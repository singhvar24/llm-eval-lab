from __future__ import annotations

import argparse
import json
import sys

from .abstain import abstention_report
from .chunking import chunk_corpus
from .retrievers import build_retriever
from .runner import Config, load_data, load_unanswerable, pareto_front, sweep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evallab", description="Evaluate RAG configurations.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep", help="evaluate the default configuration grid")
    sw.add_argument("--out", help="write full JSON results here")
    sw.add_argument("--metric", default="mrr", choices=["hit", "mrr", "ndcg", "precision", "answer_f1"])
    sw.add_argument("--model", default="medium", choices=["small", "medium", "large"])
    se = sub.add_parser("search", help="search the corpus")
    se.add_argument("query")
    se.add_argument("--retriever", default="hybrid", choices=["bm25", "tfidf", "hybrid"])
    se.add_argument("-k", type=int, default=3)
    ab = sub.add_parser("abstain", help="refusal calibration: can the top retrieval score separate unanswerable questions?")
    ab.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf", "hybrid"])
    gen = sub.add_parser("generate", help="answer one question with Claude, grounded in retrieved passages (spends tokens)")
    gen.add_argument("question")
    gen.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf", "hybrid"])
    gen.add_argument("-k", type=int, default=3)
    gen.add_argument("--model", default=None, help="default: claude-opus-5-5")
    ev = sub.add_parser("eval-llm", help="run Claude + an LLM judge over benchmark questions (spends tokens; needs --yes)")
    ev.add_argument("--limit", type=int, default=10, help="answerable questions to run (unanswerable ones are always added)")
    ev.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf", "hybrid"])
    ev.add_argument("-k", type=int, default=3)
    ev.add_argument("--model", default=None)
    ev.add_argument("--yes", action="store_true", help="confirm that real API calls will be made")
    args = p.parse_args(argv)

    corpus, qa, pricing = load_data()
    if args.cmd == "sweep":
        results = sweep(corpus, qa, pricing)
        front = {json.dumps(r["config"], sort_keys=True) for r in pareto_front(results, args.metric, args.model)}
        results.sort(key=lambda r: -r["metrics"][args.metric])
        print(f"{'chunking':<14}{'retr':<8}{'k':>2}  {'hit':>5} {'mrr':>5} {'ndcg':>5} {'f1':>5}  {'tok':>5} {'$/1k':>7}  {'ms':>6}  pareto")
        for r in results[:15]:
            c, m = r["config"], r["metrics"]
            label = c["chunking"] if c["chunking"] == "section" else f"fixed-{c['size']}/{c['overlap']}"
            star = "*" if json.dumps(c, sort_keys=True) in front else ""
            print(f"{label:<14}{c['retriever']:<8}{c['k']:>2}  {m['hit']:5.2f} {m['mrr']:5.2f} {m['ndcg']:5.2f} {m['answer_f1']:5.2f}  "
                  f"{r['avg_prompt_tokens']:5.0f} {r['cost_per_1k'][args.model]:7.3f}  {r['latency_ms']:6.2f}  {star}")
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1)
            print(f"wrote {args.out}")
    elif args.cmd == "abstain":
        rep = abstention_report(Config(retriever=args.retriever), corpus, qa, load_unanswerable())
        b = rep["best"]
        print(f"{args.retriever}: AUC {rep['auc']:.3f}; best threshold {b['threshold']:.3f} -> answers {b['coverage']:.0%} of "
              f"answerable, refuses {b['rejection']:.0%} of unanswerable")
    elif args.cmd in ("generate", "eval-llm"):
        return _llm_command(args, corpus, qa)
    else:
        chunks = chunk_corpus(corpus, "section")
        retr = build_retriever(args.retriever, chunks)
        for i, s in retr.search(args.query)[: args.k]:
            print(f"{s:7.3f}  [{chunks[i].id}] {chunks[i].text[:110]}...")
    return 0


def _passages(corpus, retriever, k, question):
    chunks = chunk_corpus(corpus, "section")
    retr = build_retriever(retriever, chunks)
    return [(chunks[i].id, chunks[i].text) for i, _ in retr.search(question)[:k]]


def _llm_command(args, corpus, qa) -> int:
    from .llm import DEFAULT_MODEL, LLMUnavailable, answer_question, estimate_cost, judge_answer, make_client

    model = args.model or DEFAULT_MODEL
    if args.cmd == "eval-llm":
        items = [dict(q, kind="answerable") for q in qa[: args.limit]] + [
            dict(u, kind="unanswerable", answer=NOT_FOUND_REF) for u in load_unanswerable()]
        print(f"Plan: {len(items)} questions x (1 answer + 1 judge call) = {2 * len(items)} real API calls on {model}.")
        if not args.yes:
            print("Re-run with --yes to proceed. (Tokens are billed to your Anthropic account.)")
            return 0
    try:
        client = make_client()
        if args.cmd == "generate":
            a = answer_question(client, args.question, _passages(corpus, args.retriever, args.k, args.question), model)
            print(a.text or "(refused)")
            print(f"citations: {a.citations}  abstained: {a.abstained}  tokens: {a.input_tokens} in / {a.output_tokens} out  "
                  f"est. ${a.cost_usd:.4f}")
            return 0
        totals = {"correct": 0, "grounded": 0, "n": 0, "cost": 0.0}
        for it in items:
            ps = _passages(corpus, args.retriever, args.k, it["question"])
            a = answer_question(client, it["question"], ps, model)
            ref = it["answer"]
            verdict, ji, jo = judge_answer(client, it["question"], ref, a.text or "(refused)", ps, model)
            totals["n"] += 1
            totals["cost"] += a.cost_usd + estimate_cost(model, ji, jo)
            if verdict:
                totals["correct"] += verdict.correct
                totals["grounded"] += verdict.grounded
            print(f"{it['id']} [{it['kind']}] correct={getattr(verdict, 'correct', None)} grounded={getattr(verdict, 'grounded', None)} abstained={a.abstained}")
        n = totals["n"]
        print(f"correct {totals['correct']}/{n}  grounded {totals['grounded']}/{n}  est. cost ${totals['cost']:.3f}")
        return 0
    except LLMUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


NOT_FOUND_REF = "The handbook does not contain this information, so the assistant should abstain."

if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
import sys

from .chunking import chunk_corpus
from .retrievers import build_retriever
from .runner import load_data, pareto_front, sweep


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
    else:
        chunks = chunk_corpus(corpus, "section")
        retr = build_retriever(args.retriever, chunks)
        for i, s in retr.search(args.query)[: args.k]:
            print(f"{s:7.3f}  [{chunks[i].id}] {chunks[i].text[:110]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

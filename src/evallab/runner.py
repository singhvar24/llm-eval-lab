from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .answerer import extractive_answer
from .chunking import chunk_corpus
from .cost import cost_per_1k
from .metrics import ndcg, reciprocal_rank, token_f1
from .retrievers import build_retriever
from .text import estimate_tokens, tokenize

# Override with EVALLAB_DATA_DIR when the package is installed outside the repository (e.g. in Docker).
DATA_DIR = Path(os.environ.get("EVALLAB_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


@dataclass(frozen=True)
class Config:
    chunking: str = "section"
    size: int = 80
    overlap: int = 20
    retriever: str = "bm25"
    k: int = 3


def load_data(data_dir: Path | None = None):
    data_dir = Path(data_dir or DATA_DIR)
    corpus = json.loads((data_dir / "corpus.json").read_text())
    qa = json.loads((data_dir / "qa.json").read_text())
    pricing = json.loads((data_dir / "pricing.json").read_text())
    return corpus, qa, pricing


def evaluate(cfg: Config, corpus, qa, pricing, measure_latency: bool = True) -> dict:
    chunks = chunk_corpus(corpus, cfg.chunking, cfg.size, cfg.overlap)
    t0 = time.perf_counter()
    retr = build_retriever(cfg.retriever, chunks)
    build_ms = (time.perf_counter() - t0) * 1000

    hit = rr = nd = prec = f1 = prompt_tokens = 0.0
    lat = 0.0
    per_question = []
    for item in qa:
        t = time.perf_counter()
        ranked = retr.search(item["question"])[: cfg.k]
        lat += (time.perf_counter() - t) * 1000
        top = [chunks[i] for i, _ in ranked]
        hits = [item["gold"] in c.sections for c in top]
        relevant_total = sum(item["gold"] in c.sections for c in chunks)
        answer = extractive_answer(item["question"], [c.text for c in top])
        q_f1 = token_f1(set(tokenize(answer)), set(tokenize(item["answer"])))
        q_prompt = pricing["system_tokens"] + estimate_tokens(item["question"]) + sum(estimate_tokens(c.text) for c in top)
        hit += any(hits)
        rr += reciprocal_rank(hits)
        nd += ndcg(hits, relevant_total)
        prec += (sum(hits) / cfg.k)
        f1 += q_f1
        prompt_tokens += q_prompt
        per_question.append({"id": item["id"], "hit": any(hits), "rank": next((i + 1 for i, h in enumerate(hits) if h), None),
                             "retrieved": [c.id for c in top], "answer_f1": round(q_f1, 4)})
    n = len(qa)
    avg_prompt = prompt_tokens / n
    result = {
        "config": asdict(cfg),
        "chunks": len(chunks),
        "metrics": {"hit": hit / n, "mrr": rr / n, "ndcg": nd / n, "precision": prec / n, "answer_f1": f1 / n},
        "avg_prompt_tokens": avg_prompt,
        "cost_per_1k": cost_per_1k(avg_prompt, pricing),
        "per_question": per_question,
    }
    if measure_latency:
        result["latency_ms"] = lat / n
        result["build_ms"] = build_ms
    return result


def default_grid() -> list[Config]:
    chunkings = [("section", 80, 20), ("fixed", 40, 10), ("fixed", 80, 20), ("fixed", 120, 30)]
    return [Config(c, s, o, r, k) for (c, s, o), r, k in itertools.product(chunkings, ("bm25", "tfidf", "hybrid"), (1, 3, 5))]


def sweep(corpus, qa, pricing, grid: list[Config] | None = None, measure_latency: bool = True) -> list[dict]:
    return [evaluate(c, corpus, qa, pricing, measure_latency) for c in (grid or default_grid())]


def pareto_front(results: list[dict], metric: str = "mrr", model: str = "medium") -> list[dict]:
    """Configs not dominated on (higher metric, lower cost)."""
    front = []
    for r in results:
        dominated = any(
            o["metrics"][metric] >= r["metrics"][metric] and o["cost_per_1k"][model] <= r["cost_per_1k"][model]
            and (o["metrics"][metric] > r["metrics"][metric] or o["cost_per_1k"][model] < r["cost_per_1k"][model])
            for o in results
        )
        if not dominated:
            front.append(r)
    return front

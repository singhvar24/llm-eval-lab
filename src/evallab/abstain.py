"""Refusal calibration: can a retrieval score tell answerable from unanswerable questions?

A grounded assistant should say "not in the documents" when nothing relevant was retrieved. This module measures
how well a simple rule — abstain when the top retrieval score is below a threshold — separates the two groups.
Raw BM25 scores are not calibrated across queries, so treat the result as a diagnostic, not a production rule.
"""
from __future__ import annotations

from .chunking import chunk_corpus
from .retrievers import build_retriever


def top_scores(cfg, corpus, questions: list[dict]) -> list[float]:
    chunks = chunk_corpus(corpus, cfg.chunking, cfg.size, cfg.overlap)
    retr = build_retriever(cfg.retriever, chunks)
    out = []
    for item in questions:
        ranked = retr.search(item["question"])
        out.append(ranked[0][1] if ranked else 0.0)
    return out


def abstention_report(cfg, corpus, answerable: list[dict], unanswerable: list[dict]) -> dict:
    a = top_scores(cfg, corpus, answerable)
    u = top_scores(cfg, corpus, unanswerable)
    thresholds = sorted(set(a) | set(u) | {0.0})
    curve = []
    for t in thresholds:
        coverage = sum(x >= t for x in a) / len(a)      # answerable questions we still answer
        rejection = sum(x < t for x in u) / len(u)      # unanswerable questions we correctly refuse
        curve.append({"threshold": t, "coverage": coverage, "rejection": rejection, "j": coverage + rejection - 1})
    best = max(curve, key=lambda p: (p["j"], -p["threshold"]))
    # AUC as the probability an answerable question scores above an unanswerable one (ties count half)
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in u)
    return {"curve": curve, "best": best, "auc": wins / (len(a) * len(u)),
            "answerable_scores": a, "unanswerable_scores": u}

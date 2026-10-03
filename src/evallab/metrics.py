from __future__ import annotations

import math


def reciprocal_rank(hits: list[bool]) -> float:
    for i, h in enumerate(hits, start=1):
        if h:
            return 1.0 / i
    return 0.0


def ndcg(hits: list[bool], total_relevant: int) -> float:
    """Binary-relevance nDCG@len(hits). ``total_relevant`` is the number of relevant chunks in the index."""
    dcg = sum(1.0 / math.log2(i + 1) for i, h in enumerate(hits, start=1) if h)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(total_relevant, len(hits)) + 1))
    return dcg / ideal if ideal else 0.0


def token_f1(pred: set[str], ref: set[str]) -> float:
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)

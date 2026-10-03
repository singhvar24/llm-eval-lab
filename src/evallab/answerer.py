"""Deterministic extractive answerer used by the offline/in-browser evaluation.

Real generation providers (Bedrock, Anthropic, ...) can implement the same ``answer`` signature; they are
intentionally not called here so the benchmark is free, reproducible and key-less.
"""
from __future__ import annotations

import re

from .text import tokenize

_SENT = re.compile(r"(?<=[.!?])\s+")


def extractive_answer(question: str, chunk_texts: list[str]) -> str:
    """Return the sentence from the top retrieved chunk that best overlaps the question."""
    if not chunk_texts:
        return ""
    q = set(tokenize(question))
    best, best_score = "", -1
    for sent in _SENT.split(chunk_texts[0]):
        score = len(q & set(tokenize(sent)))
        if score > best_score:
            best, best_score = sent, score
    return best

from __future__ import annotations

import math
from collections import Counter

from .chunking import Chunk
from .text import tokenize

RRF_K = 60


def _rank(scores: list[float]) -> list[tuple[int, float]]:
    """Indices with score > 0, best first; ties broken by index (stable and mirrored in JS)."""
    return sorted(((i, s) for i, s in enumerate(scores) if s > 0), key=lambda x: (-x[1], x[0]))


def _unique(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(tokens))


class BM25:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(t.values()) for t in self.tfs]
        self.n = len(chunks)
        self.avgdl = sum(self.lens) / self.n if self.n else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = [t for t in _unique(tokenize(query)) if t in self.idf]
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def search(self, query: str):
        return _rank(self.scores(query))


class TfIdf:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        tfs = [Counter(tokenize(c.text)) for c in chunks]
        n = len(chunks)
        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vecs = []
        self.norms = []
        for tf in tfs:
            vec = {t: (1 + math.log(f)) * self.idf[t] for t, f in tf.items()}
            self.vecs.append(vec)
            self.norms.append(math.sqrt(sum(v * v for v in vec.values())))

    def scores(self, query: str) -> list[float]:
        qtf = Counter(t for t in tokenize(query) if t in self.idf)
        qvec = {t: (1 + math.log(f)) * self.idf[t] for t, f in qtf.items()}
        qnorm = math.sqrt(sum(v * v for v in qvec.values()))
        if qnorm == 0:
            return [0.0] * len(self.vecs)
        out = []
        for vec, norm in zip(self.vecs, self.norms):
            dot = 0.0
            for t, w in qvec.items():
                if t in vec:
                    dot += w * vec[t]
            out.append(dot / (norm * qnorm) if norm else 0.0)
        return out

    def search(self, query: str):
        return _rank(self.scores(query))


class Hybrid:
    """Reciprocal Rank Fusion of BM25 and TF-IDF (Cormack et al., SIGIR 2009)."""

    name = "hybrid"

    def __init__(self, chunks: list[Chunk]):
        self.parts = (BM25(chunks), TfIdf(chunks))
        self.n = len(chunks)

    def scores(self, query: str) -> list[float]:
        fused = [0.0] * self.n
        for part in self.parts:
            for rank, (i, _) in enumerate(part.search(query), start=1):
                fused[i] += 1.0 / (RRF_K + rank)
        return fused

    def search(self, query: str):
        return _rank(self.scores(query))


def build_retriever(name: str, chunks: list[Chunk]):
    try:
        return {"bm25": BM25, "tfidf": TfIdf, "hybrid": Hybrid}[name](chunks)
    except KeyError:
        raise ValueError(f"unknown retriever: {name}") from None

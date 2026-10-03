from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    sections: tuple[str, ...] = field(default_factory=tuple)


def _section_text(sec: dict) -> str:
    return f"{sec['heading']}. {sec['text']}"


def chunk_corpus(corpus: dict, strategy: str = "section", size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split the corpus into retrievable chunks.

    ``section``: one chunk per section.
    ``fixed``: sliding word window of ``size`` words with ``overlap`` words, per document.
    A chunk records which source sections it overlaps so retrieval can be scored against gold sections.
    """
    chunks: list[Chunk] = []
    for doc in corpus["documents"]:
        if strategy == "section":
            for sec in doc["sections"]:
                chunks.append(Chunk(sec["id"], doc["id"], _section_text(sec), (sec["id"],)))
            continue
        if strategy != "fixed":
            raise ValueError(f"unknown chunking strategy: {strategy}")
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        words: list[str] = []
        tags: list[str] = []
        for sec in doc["sections"]:
            w = _section_text(sec).split()
            words += w
            tags += [sec["id"]] * len(w)
        stride = size - overlap
        start, n = 0, 0
        while start < len(words):
            end = min(start + size, len(words))
            secs = tuple(dict.fromkeys(tags[start:end]))
            chunks.append(Chunk(f"{doc['id']}-w{n}", doc["id"], " ".join(words[start:end]), secs))
            n += 1
            if end == len(words):
                break
            start += stride
    return chunks

"""Tokenisation shared (by design) with web/evallab.js — keep both in sync."""
import math
import re

_STOP_TEXT = (
    "a an and are as at be been but by can do does for from has have how i if in into is it its me my of on or our "
    "so than that the their them then there these they this to up was we were what when where which who why will with you your "
    "not no"
)
STOPWORDS = frozenset(_STOP_TEXT.split())
_WORD = re.compile(r"[a-z0-9]+")


def stem(tok: str) -> str:
    """Tiny suffix stripper (deliberately simple and mirrored in JavaScript)."""
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def raw_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def estimate_tokens(text: str) -> int:
    """Rough LLM token estimate: ~1.3 tokens per whitespace-separated word."""
    return math.ceil(len(text.split()) * 1.3)

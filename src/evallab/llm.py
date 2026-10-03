"""Grounded answering and LLM-as-judge with the Claude API (official Python SDK).

Nothing here runs unless you call it: the offline benchmark never touches the network. Every call spends real
tokens — see ``estimate_cost``. The client is injected so tests can use a fake.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-5-5"
NOT_FOUND = "NOT_IN_DOCUMENTS"
# USD per 1M tokens, Anthropic published list prices as cached 2026-09-25 — verify before relying on them.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-4-5": (1.00, 5.00)}

SYSTEM = (
    "You answer questions about a lender's mortgage-insurance handbook using ONLY the numbered context passages "
    "provided. Cite the passages you rely on by writing their ids in square brackets, for example [d03-s2]. "
    f"If the passages do not contain the answer, reply with exactly {NOT_FOUND} and nothing else. "
    "Do not use outside knowledge and do not guess. Keep answers to one or two sentences."
)
_CITE = re.compile(r"\[([a-z0-9\-]+)\]")


class LLMUnavailable(RuntimeError):
    """No usable credentials / SDK."""


def make_client():
    """Create the SDK client from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile)."""
    try:
        import anthropic

        return anthropic.Anthropic()
    except Exception as e:  # missing package or credentials
        raise LLMUnavailable(f"Claude client unavailable: {e}") from e


def _guard(call, *args, **kwargs):
    """Run an SDK call; turn the SDK's "no credentials" TypeError (raised at request time) into LLMUnavailable."""
    try:
        return call(*args, **kwargs)
    except TypeError as e:
        if "authentication" in str(e).lower():
            raise LLMUnavailable("No Claude credentials found. Set ANTHROPIC_API_KEY (or run `ant auth login`).") from e
        raise


def build_user_message(question: str, passages: list[tuple[str, str]]) -> str:
    ctx = "\n\n".join(f"[{pid}] {text}" for pid, text in passages) or "(no passages were retrieved)"
    return f"Context passages:\n\n{ctx}\n\nQuestion: {question}"


@dataclass
class Answer:
    text: str
    citations: list[str]
    abstained: bool
    refused: bool = False
    model: str = DEFAULT_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        return float("nan")
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def _text_of(response) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()


def answer_question(client, question: str, passages: list[tuple[str, str]], model: str = DEFAULT_MODEL,
                    max_tokens: int = 1024) -> Answer:
    """Ask Claude to answer from the retrieved passages. ``passages`` is a list of (chunk_id, text)."""
    response = _guard(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": build_user_message(question, passages)}],
    )
    usage = getattr(response, "usage", None)
    in_tok, out_tok = getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0
    if response.stop_reason == "refusal":
        return Answer("", [], False, True, model, in_tok, out_tok, response.stop_reason)
    text = _text_of(response)
    allowed = {pid for pid, _ in passages}
    cites = [c for c in dict.fromkeys(_CITE.findall(text)) if c in allowed]  # drop invented ids
    abstained = text.strip().strip(".").upper() == NOT_FOUND
    return Answer(text, [] if abstained else cites, abstained, False, model, in_tok, out_tok, response.stop_reason,
                  {"truncated": response.stop_reason == "max_tokens"})


class Verdict(BaseModel):
    correct: bool = Field(description="The answer agrees with the reference answer on the key fact.")
    grounded: bool = Field(description="Every claim in the answer is supported by the provided passages.")
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


JUDGE_SYSTEM = (
    "You grade answers from a retrieval-augmented assistant. Compare the candidate answer with the reference answer "
    "and check it against the context passages. If the candidate says it cannot answer, it is correct only when the "
    "reference answer is also an abstention."
)


def judge_answer(client, question: str, reference: str, candidate: str, passages: list[tuple[str, str]],
                 model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> tuple[Verdict | None, int, int]:
    """LLM-as-judge with structured output. Returns (verdict or None, input_tokens, output_tokens)."""
    prompt = (f"{build_user_message(question, passages)}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}")
    response = _guard(
        client.messages.parse,
        model=model,
        max_tokens=max_tokens,
        system=JUDGE_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Verdict,
    )
    usage = getattr(response, "usage", None)
    return (response.parsed_output, getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)

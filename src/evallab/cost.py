from __future__ import annotations


def cost_per_1k(avg_prompt_tokens: float, pricing: dict) -> dict[str, float]:
    """Estimated USD per 1,000 queries for each (illustrative) model price point."""
    out = {}
    for name, p in pricing["models"].items():
        per_query = (avg_prompt_tokens * p["in"] + pricing["answer_tokens"] * p["out"]) / 1_000_000
        out[name] = per_query * 1000
    return out

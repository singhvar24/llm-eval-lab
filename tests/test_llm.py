"""LLM paths tested with a fake client — no network, no tokens spent."""
from types import SimpleNamespace as NS

import pytest

from evallab.llm import NOT_FOUND, Verdict, answer_question, build_user_message, estimate_cost, judge_answer

PASSAGES = [("d03-s2", "The lender must lodge within 60 days."), ("d03-s3", "Assessment takes 30 business days.")]


class FakeMessages:
    def __init__(self, text="", stop="end_turn", parsed=None):
        self.calls, self.text, self.stop, self.parsed = [], text, stop, parsed

    def _resp(self):
        return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=self.text)], stop_reason=self.stop,
                  usage=NS(input_tokens=100, output_tokens=20), parsed_output=self.parsed)

    def create(self, **kw):
        self.calls.append(kw)
        return self._resp()

    parse = create


def fake(**kw):
    return NS(messages=FakeMessages(**kw))


def test_prompt_contains_ids_and_question():
    msg = build_user_message("How long?", PASSAGES)
    assert "[d03-s2]" in msg and "Question: How long?" in msg


def test_answer_keeps_valid_citations_and_drops_invented_ones():
    c = fake(text="Within 60 days [d03-s2]. Also see [d99-s9].")
    a = answer_question(c, "How long?", PASSAGES)
    assert a.citations == ["d03-s2"] and not a.abstained
    call = c.messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and "thinking" not in call and "temperature" not in call
    assert call["output_config"] == {"effort": "low"}


def test_abstention_detected():
    a = answer_question(fake(text=f"{NOT_FOUND}."), "Weather?", PASSAGES)
    assert a.abstained and a.citations == []


def test_refusal_and_truncation_flags():
    assert answer_question(fake(stop="refusal"), "q", PASSAGES).refused
    assert answer_question(fake(text="cut off", stop="max_tokens"), "q", PASSAGES).extra["truncated"]


def test_cost_estimate():
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert estimate_cost("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.0)
    assert estimate_cost("unknown-model", 1, 1) != estimate_cost("unknown-model", 1, 1)  # NaN


def test_judge_returns_structured_verdict():
    v = Verdict(correct=True, grounded=True, reasoning="Matches.")
    verdict, i, o = judge_answer(fake(parsed=v), "q", "ref", "cand", PASSAGES)
    assert verdict.correct and (i, o) == (100, 20)


def test_missing_credentials_become_llm_unavailable():
    from evallab.llm import LLMUnavailable

    class NoAuth:
        class messages:
            @staticmethod
            def create(**kw):
                raise TypeError('"Could not resolve authentication method. Expected one of api_key ..."')

    with pytest.raises(LLMUnavailable):
        answer_question(NoAuth(), "q", PASSAGES)

import math

import pytest

from evallab import Config, evaluate, load_data, pareto_front, sweep
from evallab.chunking import chunk_corpus
from evallab.metrics import ndcg, reciprocal_rank, token_f1
from evallab.retrievers import BM25, Hybrid, TfIdf
from evallab.text import estimate_tokens, tokenize

CORPUS, QA, PRICING = load_data()


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The claims were lodged") == ["claim", "lodg"]
    assert "the" not in tokenize("the policy")
    assert tokenize("Lodging") == tokenize("lodging")


def test_section_chunking_one_chunk_per_section():
    chunks = chunk_corpus(CORPUS, "section")
    assert len(chunks) == sum(len(d["sections"]) for d in CORPUS["documents"]) == 40
    assert all(len(c.sections) == 1 for c in chunks)


def test_fixed_chunking_covers_every_section_and_respects_size():
    chunks = chunk_corpus(CORPUS, "fixed", 40, 10)
    assert all(len(c.text.split()) <= 40 for c in chunks)
    covered = {s for c in chunks for s in c.sections}
    assert covered == {s["id"] for d in CORPUS["documents"] for s in d["sections"]}


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        chunk_corpus(CORPUS, "fixed", 10, 10)
    with pytest.raises(ValueError):
        chunk_corpus(CORPUS, "nope")


@pytest.mark.parametrize("cls", [BM25, TfIdf, Hybrid])
def test_retrievers_find_obvious_answer(cls):
    chunks = chunk_corpus(CORPUS, "section")
    r = cls(chunks)
    top = r.search("how many days to lodge the claim after settlement")[0][0]
    assert chunks[top].id == "d03-s2"


def test_metrics_basics():
    assert reciprocal_rank([False, True]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0
    assert ndcg([True, False, False], 1) == 1.0
    assert ndcg([False, True], 1) == pytest.approx(1 / math.log2(3))
    assert token_f1({"a", "b"}, {"b", "c"}) == pytest.approx(0.5)
    assert token_f1(set(), {"a"}) == 0.0


def test_estimate_tokens():
    assert estimate_tokens("one two three four five six seven eight nine ten") == 13


def test_evaluate_ranges_and_cost_monotonic_in_k():
    a = evaluate(Config(k=1), CORPUS, QA, PRICING)
    b = evaluate(Config(k=5), CORPUS, QA, PRICING)
    for r in (a, b):
        assert all(0 <= v <= 1 for v in r["metrics"].values())
    assert b["metrics"]["hit"] >= a["metrics"]["hit"]
    assert b["avg_prompt_tokens"] > a["avg_prompt_tokens"]
    assert b["cost_per_1k"]["medium"] > a["cost_per_1k"]["medium"]
    assert b["cost_per_1k"]["large"] > b["cost_per_1k"]["small"]


def test_pareto_front_is_non_dominated():
    results = sweep(CORPUS, QA, PRICING, measure_latency=False)
    front = pareto_front(results, "mrr", "medium")
    assert front
    for f in front:
        for o in results:
            assert not (o["metrics"]["mrr"] > f["metrics"]["mrr"] and o["cost_per_1k"]["medium"] <= f["cost_per_1k"]["medium"])


def test_abstention_report_separates_groups():
    import json

    from evallab.abstain import abstention_report
    from evallab.runner import load_unanswerable

    unans = load_unanswerable()
    assert len(unans) == 10
    rep = abstention_report(Config(retriever="bm25"), CORPUS, QA, unans)
    assert rep["auc"] > 0.75 and 0 <= rep["best"]["coverage"] <= 1
    # rank-fusion scores carry no magnitude, so they cannot separate the groups
    assert abstention_report(Config(retriever="hybrid"), CORPUS, QA, unans)["auc"] < 0.6
    assert json.dumps(rep["curve"][0])  # serialisable

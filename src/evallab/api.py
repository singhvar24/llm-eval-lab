"""FastAPI service: ``uvicorn evallab.api:app --reload``."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .abstain import abstention_report
from .chunking import chunk_corpus
from .llm import DEFAULT_MODEL, LLMUnavailable, answer_question, make_client
from .retrievers import build_retriever
from .runner import Config, default_grid, evaluate, load_data, load_unanswerable, pareto_front, sweep

app = FastAPI(title="LLM Eval Lab", version="0.1.0",
              description="Evaluate RAG configurations (retrieval quality, latency and estimated cost) over a synthetic policy corpus.")
CORPUS, QA, PRICING = load_data()
UNANSWERABLE = load_unanswerable()


class ConfigIn(BaseModel):
    chunking: str = Field("section", pattern="^(section|fixed)$")
    size: int = Field(80, ge=10, le=400)
    overlap: int = Field(20, ge=0, lt=400)
    retriever: str = Field("bm25", pattern="^(bm25|tfidf|hybrid)$")
    k: int = Field(3, ge=1, le=10)


class SearchIn(ConfigIn):
    query: str = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(CORPUS["documents"]), "questions": len(QA)}


@app.post("/search")
def search(body: SearchIn):
    cfg = body.model_dump(exclude={"query"})
    try:
        chunks = chunk_corpus(CORPUS, cfg["chunking"], cfg["size"], cfg["overlap"])
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    retr = build_retriever(cfg["retriever"], chunks)
    return {"results": [{"chunk_id": chunks[i].id, "score": s, "sections": chunks[i].sections, "text": chunks[i].text}
                        for i, s in retr.search(body.query)[: cfg["k"]]]}


@app.post("/evaluate")
def evaluate_one(body: ConfigIn):
    try:
        res = evaluate(Config(**body.model_dump()), CORPUS, QA, PRICING)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return res


@app.get("/sweep")
def run_sweep(metric: str = "mrr", model: str = "medium"):
    if metric not in {"hit", "mrr", "ndcg", "precision", "answer_f1"} or model not in PRICING["models"]:
        raise HTTPException(422, "unknown metric or model")
    results = sweep(CORPUS, QA, PRICING, default_grid())
    return {"count": len(results), "pareto": [r["config"] for r in pareto_front(results, metric, model)],
            "results": [{k: v for k, v in r.items() if k != "per_question"} for r in results]}


@app.get("/abstention")
def abstention(retriever: str = "bm25"):
    if retriever not in {"bm25", "tfidf", "hybrid"}:
        raise HTTPException(422, "unknown retriever")
    rep = abstention_report(Config(retriever=retriever), CORPUS, QA, UNANSWERABLE)
    return {"auc": rep["auc"], "best": rep["best"], "curve": rep["curve"]}


def get_llm_client():
    try:
        return make_client()
    except LLMUnavailable as e:
        raise HTTPException(503, str(e)) from e


class AnswerIn(SearchIn):
    model: str = Field(DEFAULT_MODEL, max_length=64)


@app.post("/answer")
def answer(body: AnswerIn, client=Depends(get_llm_client)):
    """Retrieve passages, then ask Claude to answer from them only. Spends tokens on the server's credentials."""
    chunks = chunk_corpus(CORPUS, "section")
    retr = build_retriever(body.retriever, chunks)
    ranked = retr.search(body.query)[: body.k]
    passages = [(chunks[i].id, chunks[i].text) for i, _ in ranked]
    try:
        a = answer_question(client, body.query, passages, body.model)
    except LLMUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:  # SDK errors: auth, rate limit, connection ...
        status = getattr(e, "status_code", None)
        raise HTTPException(503 if status in (401, 403) else 502, f"Claude API error: {type(e).__name__}") from e
    return {"answer": a.text, "citations": a.citations, "abstained": a.abstained, "refused": a.refused, "model": a.model,
            "usage": {"input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "est_cost_usd": a.cost_usd},
            "passages": [{"chunk_id": pid} for pid, _ in passages]}

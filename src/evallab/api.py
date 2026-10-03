"""FastAPI service: ``uvicorn evallab.api:app --reload``."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chunking import chunk_corpus
from .retrievers import build_retriever
from .runner import Config, default_grid, evaluate, load_data, pareto_front, sweep

app = FastAPI(title="LLM Eval Lab", version="0.1.0",
              description="Evaluate RAG configurations (retrieval quality, latency and estimated cost) over a synthetic policy corpus.")
CORPUS, QA, PRICING = load_data()


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

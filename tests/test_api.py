from fastapi.testclient import TestClient

from evallab.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["questions"] == 32


def test_search():
    r = client.post("/search", json={"query": "how long are records kept", "retriever": "bm25", "k": 2})
    assert r.status_code == 200
    assert r.json()["results"][0]["chunk_id"] == "d07-s3"


def test_evaluate_and_validation():
    ok = client.post("/evaluate", json={"retriever": "hybrid", "k": 3})
    assert ok.status_code == 200 and 0 <= ok.json()["metrics"]["mrr"] <= 1
    assert client.post("/evaluate", json={"retriever": "magic"}).status_code == 422
    assert client.post("/evaluate", json={"chunking": "fixed", "size": 20, "overlap": 30}).status_code == 422


def test_sweep():
    r = client.get("/sweep?metric=mrr&model=small")
    assert r.status_code == 200 and r.json()["count"] == 36 and r.json()["pareto"]
    assert client.get("/sweep?metric=bogus").status_code == 422

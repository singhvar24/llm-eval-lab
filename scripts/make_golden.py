"""Write tests/golden/metrics.json from the Python implementation (latency excluded)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evallab.abstain import abstention_report
from evallab.runner import Config, default_grid, evaluate, load_data, load_unanswerable

corpus, qa, pricing = load_data()
out = []
for cfg in default_grid():
    r = evaluate(cfg, corpus, qa, pricing, measure_latency=False)
    out.append({"config": r["config"], "chunks": r["chunks"], "metrics": r["metrics"], "avg_prompt_tokens": r["avg_prompt_tokens"],
                "cost_per_1k": r["cost_per_1k"], "hits": [q["hit"] for q in r["per_question"]], "ranks": [q["rank"] for q in r["per_question"]]})
(ROOT / "tests" / "golden" / "metrics.json").write_text(json.dumps(out))
print("wrote", len(out), "golden configs")

unans = load_unanswerable()
ab = {}
for name in ("bm25", "tfidf", "hybrid"):
    rep = abstention_report(Config(retriever=name), corpus, qa, unans)
    ab[name] = {"auc": rep["auc"], "best": rep["best"], "points": len(rep["curve"])}
(ROOT / "tests" / "golden" / "abstain.json").write_text(json.dumps(ab))
print("wrote abstention golden for", ", ".join(ab))

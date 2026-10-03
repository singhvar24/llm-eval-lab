"""Write tests/golden/metrics.json from the Python implementation (latency excluded)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evallab.runner import default_grid, evaluate, load_data

corpus, qa, pricing = load_data()
out = []
for cfg in default_grid():
    r = evaluate(cfg, corpus, qa, pricing, measure_latency=False)
    out.append({"config": r["config"], "chunks": r["chunks"], "metrics": r["metrics"], "avg_prompt_tokens": r["avg_prompt_tokens"],
                "cost_per_1k": r["cost_per_1k"], "hits": [q["hit"] for q in r["per_question"]], "ranks": [q["rank"] for q in r["per_question"]]})
(ROOT / "tests" / "golden" / "metrics.json").write_text(json.dumps(out))
print("wrote", len(out), "golden configs")

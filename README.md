# LLM Eval Lab

**Live demo:** https://singhvar24.github.io/llm-eval-lab/ · **Portfolio:** https://singhvar24.github.io/

[![Cost vs quality chart with Pareto frontier](preview.jpg)](https://singhvar24.github.io/llm-eval-lab/)

A small, honest toolkit for answering a practical RAG question: **which retrieval configuration is worth its cost?**
It evaluates chunking strategy × retriever × top-k on retrieval quality, estimated prompt cost and measured latency, and shows the Pareto frontier.

![stack](https://img.shields.io/badge/Python-3.10%2B-blue) ![stack](https://img.shields.io/badge/FastAPI-API-009688) ![stack](https://img.shields.io/badge/JavaScript-ES%20modules-f7df1e) ![license](https://img.shields.io/badge/license-MIT-green)

## What is in the box

| Part | Path | Notes |
|---|---|---|
| Python toolkit | `src/evallab/` | Chunking, BM25, TF-IDF cosine, hybrid (reciprocal-rank fusion), metrics, cost model, CLI |
| REST API | `src/evallab/api.py` | FastAPI: `/health`, `/search`, `/evaluate`, `/sweep` (validated inputs, OpenAPI docs at `/docs`) |
| In-browser demo | `index.html`, `web/` | Vanilla JS port of the toolkit, interactive trade-off chart, question inspector, free-text search |
| Dataset | `data/` | Synthetic mortgage-insurance handbook (8 documents, 40 sections), 32 questions with gold sections |
| Tests / CI | `tests/`, `.github/workflows/ci.yml` | pytest (15), Node parity test (5), ruff, data-regeneration check |
| Container | `Dockerfile` | Runs the API with uvicorn |

```
data/*.json ──► chunking ──► retriever (BM25 | TF-IDF | RRF hybrid) ──► top-k chunks
                                                            │
                          metrics: Hit@k, MRR, nDCG@k, Precision@k   │
                          extractive answer ► Answer F1              ▼
                          prompt tokens ► est. cost / 1k queries     Pareto frontier
```

## Quick start

```bash
pip install -e ".[dev]"
evallab sweep                 # table of the best configurations (add --out results.json)
evallab search "how long are records kept" --retriever hybrid -k 3
uvicorn evallab.api:app --reload      # http://127.0.0.1:8000/docs
pytest -q && node --test tests/parity.test.mjs
python -m http.server 8000    # then open http://localhost:8000/ for the demo
```

## Method

* **Chunking:** per section, or sliding word windows (40/80/120 words, 25 % overlap). A chunk is *relevant* when it overlaps the gold section.
* **Retrievers:** BM25 (k1 = 1.5, b = 0.75); TF-IDF cosine with sublinear tf; hybrid = reciprocal-rank fusion (k = 60; Cormack et al., SIGIR 2009).
* **Metrics:** Hit@k, MRR@k, binary-relevance nDCG@k, Precision@k, and token-overlap F1 of an extractive answer against a reference answer.
* **Cost:** prompt tokens = system (60) + question + retrieved chunks, using ~1.3 tokens per word; output fixed at 80 tokens. Prices in `data/pricing.json` are **illustrative placeholders, not real vendor prices** — edit them.
* **Latency:** mean wall-clock retrieval time per query on the machine running it.

## Limitations (please read)

* The corpus and questions are **synthetic and small** (32 questions). Many configurations land within a few points of each other, so differences are indicative, not statistically significant.
* **No LLM is called.** Answer quality comes from a deterministic extractive baseline, so "Answer F1" measures retrieval + sentence selection, not generation. The `extractive_answer` function is the intended plug-in point for a Bedrock or Anthropic-API generator; that integration is **not implemented here**.
* The fictional insurer and all figures in the corpus are invented and do not describe any real product or company.
* `Dockerfile` and `.github/workflows/ci.yml` were written and the commands inside them were run locally, but the image build and the GitHub Actions run had not been executed locally; the Actions workflow has since passed on GitHub.

## Parity between Python and the browser

`web/evallab.js` re-implements the Python code. `scripts/make_golden.py` writes the Python results for all 36 configurations to `tests/golden/metrics.json`, and `tests/parity.test.mjs` asserts the JavaScript reproduces them (metrics to 1e-9, plus per-question hits and ranks).

## Publishing the demo

Settings → Pages → *Deploy from a branch* → `main` / `(root)`. The site is plain static files; no build step.

GitHub Actions CI (pytest, Node parity test, ruff) passed on the first run.

## Author

Built by [Varnika Singh](https://singhvar24.github.io/) — Graduate AI Engineer, Sydney. MIT licensed.

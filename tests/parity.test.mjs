// Verifies the JavaScript port reproduces the Python results (run: node --test tests/)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { evaluate, defaultGrid, tokenize, chunkCorpus, paretoFront, buildRetriever } from "../web/evallab.js";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const corpus = load("../data/corpus.json"), qa = load("../data/qa.json"), pricing = load("../data/pricing.json");
const golden = load("./golden/metrics.json");
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, `${msg}: ${a} vs ${b}`);

test("JS metrics match Python for all 36 configurations", () => {
  assert.equal(golden.length, defaultGrid().length);
  golden.forEach((g) => {
    const r = evaluate(g.config, corpus, qa, pricing);
    const tag = JSON.stringify(g.config);
    assert.equal(r.chunks, g.chunks, tag);
    for (const m of Object.keys(g.metrics)) close(r.metrics[m], g.metrics[m], `${tag} ${m}`);
    close(r.avg_prompt_tokens, g.avg_prompt_tokens, `${tag} tokens`);
    for (const m of Object.keys(g.cost_per_1k)) close(r.cost_per_1k[m], g.cost_per_1k[m], `${tag} cost ${m}`);
    assert.deepEqual(r.per_question.map((q) => q.hit), g.hits, `${tag} hits`);
    assert.deepEqual(r.per_question.map((q) => q.rank), g.ranks, `${tag} ranks`);
  });
});

test("tokeniser matches the documented behaviour", () => {
  assert.deepEqual(tokenize("The claims were lodged"), ["claim", "lodg"]);
});

test("retrievers find the obvious answer", () => {
  const chunks = chunkCorpus(corpus, "section");
  for (const name of ["bm25", "tfidf", "hybrid"]) {
    const top = buildRetriever(name, chunks).search("how many days to lodge the claim after settlement")[0][0];
    assert.equal(chunks[top].id, "d03-s2", name);
  }
});

test("invalid chunking parameters throw", () => {
  assert.throws(() => chunkCorpus(corpus, "fixed", 10, 10));
  assert.throws(() => chunkCorpus(corpus, "bogus"));
});

test("pareto front is non-empty and non-dominated", () => {
  const results = defaultGrid().map((c) => evaluate(c, corpus, qa, pricing));
  const front = paretoFront(results);
  assert.ok(front.length > 0);
  for (const f of front) for (const o of results)
    assert.ok(!(o.metrics.mrr > f.metrics.mrr && o.cost_per_1k.medium <= f.cost_per_1k.medium));
});

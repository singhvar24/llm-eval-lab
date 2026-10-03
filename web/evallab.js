// Browser/Node port of the Python `evallab` package. No DOM access here, so Node can test it.
// Tokenisation, ranking and metric definitions mirror src/evallab/*.py — a parity test enforces this.

export const STOPWORDS = new Set(
  ("a an and are as at be been but by can do does for from has have how i if in into is it its me my of on or our " +
   "so than that the their them then there these they this to up was we were what when where which who why will with you your " +
   "not no").split(" "));

export function stem(t) {
  if (t.length > 5 && t.endsWith("ing")) return t.slice(0, -3);
  if (t.length > 4 && t.endsWith("ed")) return t.slice(0, -2);
  if (t.length > 4 && t.endsWith("es")) return t.slice(0, -2);
  if (t.length > 3 && t.endsWith("s") && !t.endsWith("ss")) return t.slice(0, -1);
  return t;
}
export const rawTokens = (text) => text.toLowerCase().match(/[a-z0-9]+/g) || [];
export const tokenize = (text) => rawTokens(text).filter((t) => !STOPWORDS.has(t)).map(stem);
export const words = (text) => text.split(/\s+/).filter(Boolean);
export const estimateTokens = (text) => Math.ceil(words(text).length * 1.3);
const unique = (a) => [...new Set(a)];
const counter = (a) => { const m = new Map(); for (const t of a) m.set(t, (m.get(t) || 0) + 1); return m; };

// ---------- chunking ----------
const sectionText = (s) => `${s.heading}. ${s.text}`;

export function chunkCorpus(corpus, strategy = "section", size = 80, overlap = 20) {
  const chunks = [];
  for (const doc of corpus.documents) {
    if (strategy === "section") {
      for (const s of doc.sections) chunks.push({ id: s.id, docId: doc.id, text: sectionText(s), sections: [s.id] });
      continue;
    }
    if (strategy !== "fixed") throw new Error(`unknown chunking strategy: ${strategy}`);
    if (size <= 0 || overlap < 0 || overlap >= size) throw new Error("need size > 0 and 0 <= overlap < size");
    const ws = [], tags = [];
    for (const s of doc.sections) { const w = words(sectionText(s)); for (const x of w) { ws.push(x); tags.push(s.id); } }
    const stride = size - overlap;
    let start = 0, n = 0;
    while (start < ws.length) {
      const end = Math.min(start + size, ws.length);
      chunks.push({ id: `${doc.id}-w${n}`, docId: doc.id, text: ws.slice(start, end).join(" "), sections: unique(tags.slice(start, end)) });
      n++;
      if (end === ws.length) break;
      start += stride;
    }
  }
  return chunks;
}

// ---------- retrieval ----------
const RRF_K = 60;
function rank(scores) {
  const out = [];
  scores.forEach((s, i) => { if (s > 0) out.push([i, s]); });
  return out.sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]));
}

export class BM25 {
  constructor(chunks, k1 = 1.5, b = 0.75) {
    this.k1 = k1; this.b = b;
    this.tfs = chunks.map((c) => counter(tokenize(c.text)));
    this.lens = this.tfs.map((tf) => { let s = 0; for (const v of tf.values()) s += v; return s; });
    this.n = chunks.length;
    this.avgdl = this.n ? this.lens.reduce((a, b2) => a + b2, 0) / this.n : 0;
    const df = new Map();
    for (const tf of this.tfs) for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    this.idf = new Map();
    for (const [t, d] of df) this.idf.set(t, Math.log(1 + (this.n - d + 0.5) / (d + 0.5)));
  }
  scores(query) {
    const q = unique(tokenize(query)).filter((t) => this.idf.has(t));
    return this.tfs.map((tf, i) => {
      let s = 0;
      for (const t of q) {
        const f = tf.get(t) || 0;
        if (f) s += this.idf.get(t) * f * (this.k1 + 1) / (f + this.k1 * (1 - this.b + this.b * this.lens[i] / this.avgdl));
      }
      return s;
    });
  }
  search(query) { return rank(this.scores(query)); }
}

export class TfIdf {
  constructor(chunks) {
    const tfs = chunks.map((c) => counter(tokenize(c.text)));
    const n = chunks.length;
    const df = new Map();
    for (const tf of tfs) for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    this.idf = new Map();
    for (const [t, d] of df) this.idf.set(t, Math.log((1 + n) / (1 + d)) + 1);
    this.vecs = []; this.norms = [];
    for (const tf of tfs) {
      const vec = new Map();
      for (const [t, f] of tf) vec.set(t, (1 + Math.log(f)) * this.idf.get(t));
      let s = 0; for (const v of vec.values()) s += v * v;
      this.vecs.push(vec); this.norms.push(Math.sqrt(s));
    }
  }
  scores(query) {
    const qtf = counter(tokenize(query).filter((t) => this.idf.has(t)));
    const qvec = new Map();
    for (const [t, f] of qtf) qvec.set(t, (1 + Math.log(f)) * this.idf.get(t));
    let qs = 0; for (const v of qvec.values()) qs += v * v;
    const qnorm = Math.sqrt(qs);
    if (qnorm === 0) return this.vecs.map(() => 0);
    return this.vecs.map((vec, i) => {
      let dot = 0;
      for (const [t, w] of qvec) if (vec.has(t)) dot += w * vec.get(t);
      return this.norms[i] ? dot / (this.norms[i] * qnorm) : 0;
    });
  }
  search(query) { return rank(this.scores(query)); }
}

export class Hybrid {
  constructor(chunks) { this.parts = [new BM25(chunks), new TfIdf(chunks)]; this.n = chunks.length; }
  scores(query) {
    const fused = new Array(this.n).fill(0);
    for (const p of this.parts) p.search(query).forEach(([i], r) => { fused[i] += 1 / (RRF_K + r + 1); });
    return fused;
  }
  search(query) { return rank(this.scores(query)); }
}

export function buildRetriever(name, chunks) {
  const R = { bm25: BM25, tfidf: TfIdf, hybrid: Hybrid }[name];
  if (!R) throw new Error(`unknown retriever: ${name}`);
  return new R(chunks);
}

// ---------- metrics / answering / cost ----------
export function reciprocalRank(hits) { const i = hits.indexOf(true); return i < 0 ? 0 : 1 / (i + 1); }
export function ndcg(hits, totalRelevant) {
  let dcg = 0; hits.forEach((h, i) => { if (h) dcg += 1 / Math.log2(i + 2); });
  let ideal = 0; for (let i = 1; i <= Math.min(totalRelevant, hits.length); i++) ideal += 1 / Math.log2(i + 1);
  return ideal ? dcg / ideal : 0;
}
export function tokenF1(pred, ref) {
  if (!pred.size || !ref.size) return 0;
  let o = 0; for (const t of pred) if (ref.has(t)) o++;
  if (!o) return 0;
  const p = o / pred.size, r = o / ref.size;
  return 2 * p * r / (p + r);
}
export function extractiveAnswer(question, chunkTexts) {
  if (!chunkTexts.length) return "";
  const q = new Set(tokenize(question));
  let best = "", bestScore = -1;
  for (const sent of chunkTexts[0].split(/(?<=[.!?])\s+/)) {
    let score = 0; for (const t of new Set(tokenize(sent))) if (q.has(t)) score++;
    if (score > bestScore) { best = sent; bestScore = score; }
  }
  return best;
}
export function costPer1k(avgPrompt, pricing) {
  const out = {};
  for (const [name, p] of Object.entries(pricing.models))
    out[name] = ((avgPrompt * p.in + pricing.answer_tokens * p.out) / 1e6) * 1000;
  return out;
}

// ---------- evaluation ----------
export const defaultConfig = { chunking: "section", size: 80, overlap: 20, retriever: "bm25", k: 3 };

export function evaluate(cfg, corpus, qa, pricing, now = () => performance.now()) {
  const chunks = chunkCorpus(corpus, cfg.chunking, cfg.size, cfg.overlap);
  const t0 = now();
  const retr = buildRetriever(cfg.retriever, chunks);
  const buildMs = now() - t0;
  let hit = 0, rr = 0, nd = 0, prec = 0, f1 = 0, promptTokens = 0, lat = 0;
  const perQuestion = [];
  for (const item of qa) {
    const t = now();
    const ranked = retr.search(item.question).slice(0, cfg.k);
    lat += now() - t;
    const top = ranked.map(([i]) => chunks[i]);
    const hits = top.map((c) => c.sections.includes(item.gold));
    const relevantTotal = chunks.filter((c) => c.sections.includes(item.gold)).length;
    const answer = extractiveAnswer(item.question, top.map((c) => c.text));
    const qF1 = tokenF1(new Set(tokenize(answer)), new Set(tokenize(item.answer)));
    const qPrompt = pricing.system_tokens + estimateTokens(item.question) + top.reduce((s, c) => s + estimateTokens(c.text), 0);
    const any = hits.some(Boolean);
    hit += any ? 1 : 0; rr += reciprocalRank(hits); nd += ndcg(hits, relevantTotal);
    prec += hits.filter(Boolean).length / cfg.k; f1 += qF1; promptTokens += qPrompt;
    const r = hits.indexOf(true);
    perQuestion.push({ id: item.id, hit: any, rank: r < 0 ? null : r + 1, retrieved: top.map((c) => c.id), answer, answer_f1: qF1 });
  }
  const n = qa.length, avgPrompt = promptTokens / n;
  return {
    config: { ...cfg }, chunks: chunks.length,
    metrics: { hit: hit / n, mrr: rr / n, ndcg: nd / n, precision: prec / n, answer_f1: f1 / n },
    avg_prompt_tokens: avgPrompt, cost_per_1k: costPer1k(avgPrompt, pricing),
    latency_ms: lat / n, build_ms: buildMs, per_question: perQuestion,
  };
}

export function defaultGrid() {
  const chunkings = [["section", 80, 20], ["fixed", 40, 10], ["fixed", 80, 20], ["fixed", 120, 30]];
  const grid = [];
  for (const [chunking, size, overlap] of chunkings)
    for (const retriever of ["bm25", "tfidf", "hybrid"]) for (const k of [1, 3, 5]) grid.push({ chunking, size, overlap, retriever, k });
  return grid;
}

export function paretoFront(results, metric = "mrr", model = "medium") {
  return results.filter((r) => !results.some((o) =>
    o.metrics[metric] >= r.metrics[metric] && o.cost_per_1k[model] <= r.cost_per_1k[model] &&
    (o.metrics[metric] > r.metrics[metric] || o.cost_per_1k[model] < r.cost_per_1k[model])));
}

// ---------- refusal calibration (mirror of src/evallab/abstain.py) ----------
export function topScores(cfg, corpus, questions) {
  const chunks = chunkCorpus(corpus, cfg.chunking, cfg.size, cfg.overlap);
  const retr = buildRetriever(cfg.retriever, chunks);
  return questions.map((q) => { const r = retr.search(q.question); return r.length ? r[0][1] : 0; });
}

export function abstentionReport(cfg, corpus, answerable, unanswerable) {
  const a = topScores(cfg, corpus, answerable), u = topScores(cfg, corpus, unanswerable);
  const thresholds = [...new Set([...a, ...u, 0])].sort((x, y) => x - y);
  const curve = thresholds.map((t) => {
    const coverage = a.filter((x) => x >= t).length / a.length;
    const rejection = u.filter((x) => x < t).length / u.length;
    return { threshold: t, coverage, rejection, j: coverage + rejection - 1 };
  });
  let best = curve[0];
  for (const p of curve) if (p.j > best.j || (p.j === best.j && p.threshold < best.threshold)) best = p;
  let wins = 0;
  for (const x of a) for (const y of u) wins += (x > y ? 1 : 0) + (x === y ? 0.5 : 0);
  return { curve, best, auc: wins / (a.length * u.length), answerable_scores: a, unanswerable_scores: u };
}

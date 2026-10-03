import { evaluate, defaultGrid, paretoFront, chunkCorpus, buildRetriever, extractiveAnswer } from "./evallab.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const COLORS = { bm25: "#0d9488", tfidf: "#6366f1", hybrid: "#f59e0b" };
const NAMES = { bm25: "BM25", tfidf: "TF-IDF", hybrid: "Hybrid (RRF)" };
const METRIC_LABEL = { mrr: "MRR", hit: "Hit@k", ndcg: "nDCG@k", precision: "Precision@k", answer_f1: "Answer F1" };

let corpus, qa, pricing, all = [], byKey = new Map();
const key = (c) => `${c.chunking}|${c.size}|${c.overlap}|${c.retriever}|${c.k}`;
const label = (c) => (c.chunking === "section" ? "section" : `fixed ${c.size}/${c.overlap}`);
const state = { retriever: "bm25", chunking: "section|80|20", k: 3, model: "medium", metric: "mrr", sort: { col: "metric", dir: -1 } };

const cfgFromState = () => { const [chunking, size, overlap] = state.chunking.split("|"); return { chunking, size: +size, overlap: +overlap, retriever: state.retriever, k: state.k }; };
const fmt = (x, d = 2) => x.toFixed(d);

async function load() {
  const j = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
  [corpus, qa, pricing] = await Promise.all([j("data/corpus.json"), j("data/qa.json"), j("data/pricing.json")]);
}

function syncControls() {
  document.querySelector(`input[name=retriever][value=${state.retriever}]`).checked = true;
  document.querySelector("[name=chunking]").value = state.chunking;
  document.querySelector("[name=k]").value = state.k; $("#k-out").textContent = state.k;
  document.querySelector("[name=model]").value = state.model;
}

function renderCards(r) {
  const m = r.metrics, c = r.cost_per_1k[state.model];
  const card = (name, val, sub = "") => `<div class="card"><span>${name}</span><b>${val}</b><small>${sub}</small></div>`;
  $("#cards").innerHTML =
    card(`Hit@${r.config.k}`, fmt(m.hit), "gold section retrieved") + card("MRR", fmt(m.mrr), "rank of first hit") + card(`nDCG@${r.config.k}`, fmt(m.ndcg), `${r.chunks} chunks indexed`) +
    card("Answer F1", fmt(m.answer_f1), "extractive baseline") + card("Prompt tokens", Math.round(r.avg_prompt_tokens), "avg per query (est.)") +
    card("Cost / 1k queries", "$" + fmt(c, 2), `${state.model} price model`) + card("Latency", fmt(r.latency_ms, 3) + " ms", "retrieval, measured here");
}

function current() { return evaluate(cfgFromState(), corpus, qa, pricing); }

function renderChart(front) {
  const W = 720, H = 360, L = 56, R = 16, T = 16, B = 44;
  const xs = all.map((r) => r.cost_per_1k[state.model]), ys = all.map((r) => r.metrics[state.metric]);
  const x0 = Math.min(...xs) * 0.95, x1 = Math.max(...xs) * 1.03;
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = Math.max((y1 - y0) * 0.15, 0.02); y0 = Math.max(0, y0 - pad); y1 = Math.min(1, y1 + pad);
  const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const sel = key(cfgFromState());
  let s = `<title id="chart-t">Quality versus estimated cost</title><desc id="chart-d">Each point is a configuration; higher and further left is better. Line shows the Pareto frontier.</desc>`;
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4, xv = x0 + ((x1 - x0) * i) / 4;
    s += `<line x1="${L}" x2="${W - R}" y1="${Y(yv)}" y2="${Y(yv)}" stroke="var(--line)"/><text x="${L - 8}" y="${Y(yv) + 4}" text-anchor="end">${fmt(yv)}</text>`;
    s += `<text x="${X(xv)}" y="${H - B + 18}" text-anchor="middle">$${fmt(xv, 1)}</text>`;
  }
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">Estimated cost per 1,000 queries (${state.model} price model, illustrative) →</text>`;
  s += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${METRIC_LABEL[state.metric]} →</text>`;
  const f = [...front].sort((a, b) => a.cost_per_1k[state.model] - b.cost_per_1k[state.model]);
  s += `<polyline fill="none" stroke="var(--ink)" stroke-dasharray="5 4" stroke-width="1.5" opacity=".55" points="${f.map((r) => `${X(r.cost_per_1k[state.model])},${Y(r.metrics[state.metric])}`).join(" ")}"/>`;
  all.forEach((r) => {
    const k = key(r.config), cx = X(r.cost_per_1k[state.model]), cy = Y(r.metrics[state.metric]);
    const onFront = front.includes(r);
    s += `<circle class="pt${k === sel ? " sel" : ""}" data-k="${esc(k)}" cx="${cx}" cy="${cy}" r="${onFront ? 8 : 6}" fill="${COLORS[r.config.retriever]}" tabindex="0" role="button" ` +
      `aria-label="${esc(NAMES[r.config.retriever])}, ${label(r.config)}, k=${r.config.k}: ${METRIC_LABEL[state.metric]} ${fmt(r.metrics[state.metric])}, cost $${fmt(r.cost_per_1k[state.model], 2)}${onFront ? ", on Pareto frontier" : ""}">` +
      `<title>${esc(NAMES[r.config.retriever])} · ${label(r.config)} · k=${r.config.k}\n${METRIC_LABEL[state.metric]} ${fmt(r.metrics[state.metric])} · $${fmt(r.cost_per_1k[state.model], 2)}/1k</title></circle>`;
  });
  $("#chart").innerHTML = s;
  $("#legend").innerHTML = Object.entries(NAMES).map(([k, n]) => `<span><i style="background:${COLORS[k]}"></i>${n}</span>`).join("") + `<span>Large dot = on Pareto frontier</span>`;
}

function renderTable(front) {
  const m = state.metric, model = state.model, sel = key(cfgFromState());
  const val = { metric: (r) => r.metrics[m], cost: (r) => r.cost_per_1k[model], latency: (r) => r.latency_ms, tokens: (r) => r.avg_prompt_tokens, f1: (r) => r.metrics.answer_f1 };
  const rows = [...all].sort((a, b) => (val[state.sort.col](a) - val[state.sort.col](b)) * state.sort.dir).slice(0, 12);
  const th = (col, text) => `<th data-col="${col}" scope="col" aria-sort="${state.sort.col === col ? (state.sort.dir < 0 ? "descending" : "ascending") : "none"}">${text}${state.sort.col === col ? (state.sort.dir < 0 ? " ▼" : " ▲") : ""}</th>`;
  $("#table").innerHTML = `<thead><tr><th scope="col">Chunking</th><th scope="col">Retriever</th><th scope="col">k</th>${th("metric", METRIC_LABEL[m])}${th("f1", "Answer F1")}${th("tokens", "Tokens")}${th("cost", "$/1k")}${th("latency", "ms/query")}<th scope="col">Pareto</th></tr></thead><tbody>` +
    rows.map((r) => `<tr data-k="${esc(key(r.config))}" class="${key(r.config) === sel ? "sel" : ""}" tabindex="0"><td>${label(r.config)}</td><td>${NAMES[r.config.retriever]}</td><td>${r.config.k}</td><td>${fmt(r.metrics[m])}</td><td>${fmt(r.metrics.answer_f1)}</td><td>${Math.round(r.avg_prompt_tokens)}</td><td>$${fmt(r.cost_per_1k[model], 2)}</td><td>${fmt(r.latency_ms, 3)}</td><td class="star">${front.includes(r) ? "★" : ""}</td></tr>`).join("") + "</tbody>";
}

function chunkHtml(c, rank, gold, score) {
  const hit = c.sections.includes(gold);
  return `<div class="chunk${hit ? " gold" : ""}"><h3>#${rank} <span class="tag">${esc(c.id)}</span>${gold ? `<span class="tag ${hit ? "ok" : "no"}">${hit ? "contains gold section" : "not relevant"}</span>` : ""}${score != null ? `<span class="tag">score ${score.toFixed(3)}</span>` : ""}</h3><p>${esc(c.text)}</p></div>`;
}

function renderInspector() {
  const item = qa.find((q) => q.id === $("#question").value), cfg = cfgFromState();
  const chunks = chunkCorpus(corpus, cfg.chunking, cfg.size, cfg.overlap), retr = buildRetriever(cfg.retriever, chunks);
  const ranked = retr.search(item.question).slice(0, cfg.k), top = ranked.map(([i]) => chunks[i]);
  const answer = extractiveAnswer(item.question, top.map((c) => c.text));
  const hit = top.some((c) => c.sections.includes(item.gold));
  $("#inspect").innerHTML = `<div class="answer"><strong>Gold section:</strong> ${esc(item.gold)} · <strong>Reference answer:</strong> ${esc(item.answer)}<br><strong>Result:</strong> <span class="tag ${hit ? "ok" : "no"}">${hit ? "retrieved" : "missed"}</span> with ${esc(NAMES[cfg.retriever])}, ${label(cfg)}, k=${cfg.k}<br><strong>Extractive answer:</strong> ${esc(answer || "—")}</div>` +
    (top.length ? ranked.map(([i, s], r) => chunkHtml(chunks[i], r + 1, item.gold, s)).join("") : `<div class="chunk"><p>No chunk matched any query term.</p></div>`);
}

function renderPlayground(q) {
  const cfg = cfgFromState(), chunks = chunkCorpus(corpus, cfg.chunking, cfg.size, cfg.overlap), retr = buildRetriever(cfg.retriever, chunks);
  const ranked = retr.search(q).slice(0, cfg.k);
  if (!ranked.length) { $("#play-out").innerHTML = `<div class="chunk"><p>No matches. Try different keywords (the corpus is about mortgage insurance policies, claims, hardship, privacy and complaints).</p></div>`; return; }
  const answer = extractiveAnswer(q, ranked.map(([i]) => chunks[i].text));
  $("#play-out").innerHTML = `<div class="answer"><strong>Extractive answer:</strong> ${esc(answer)} <em>[${esc(chunks[ranked[0][0]].id)}]</em></div>` + ranked.map(([i, s], r) => chunkHtml(chunks[i], r + 1, null, s)).join("");
}

function refresh() {
  const r = current(); byKey.set(key(r.config), r);
  renderCards(r);
  const front = paretoFront(all, state.metric, state.model);
  renderChart(front); renderTable(front); renderInspector();
  const q = $("#q").value.trim(); if (q) renderPlayground(q);
}

function pick(k) {
  const r = byKey.get(k); if (!r) return;
  state.retriever = r.config.retriever; state.chunking = `${r.config.chunking}|${r.config.size}|${r.config.overlap}`; state.k = r.config.k;
  syncControls(); refresh();
}

async function init() {
  await load();
  all = defaultGrid().map((c) => evaluate(c, corpus, qa, pricing)); all.forEach((r) => byKey.set(key(r.config), r));
  $("#question").innerHTML = qa.map((q) => `<option value="${q.id}">${esc(q.question)}</option>`).join("");
  $("#controls").addEventListener("input", (e) => {
    const n = e.target.name;
    if (n === "retriever") state.retriever = e.target.value; else if (n === "chunking") state.chunking = e.target.value;
    else if (n === "k") { state.k = +e.target.value; $("#k-out").textContent = state.k; } else if (n === "model") state.model = e.target.value;
    refresh();
  });
  $("#metric").addEventListener("change", (e) => { state.metric = e.target.value; refresh(); });
  $("#question").addEventListener("change", renderInspector);
  $("#play").addEventListener("submit", () => { const q = $("#q").value.trim(); if (q) renderPlayground(q); });
  $("#chart").addEventListener("click", (e) => { const k = e.target.closest?.("[data-k]")?.dataset.k; if (k) pick(k); });
  $("#chart").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { const k = e.target.dataset?.k; if (k) { e.preventDefault(); pick(k); } } });
  $("#table").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-col]"); if (th) { const c = th.dataset.col; state.sort = { col: c, dir: state.sort.col === c ? -state.sort.dir : (c === "cost" || c === "latency" || c === "tokens" ? 1 : -1) }; refresh(); return; }
    const tr = e.target.closest("tr[data-k]"); if (tr) pick(tr.dataset.k);
  });
  $("#table").addEventListener("keydown", (e) => { if (e.key === "Enter") { const tr = e.target.closest("tr[data-k]"); if (tr) pick(tr.dataset.k); } });
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
  });
  refresh();
}
init().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="notice" role="alert">Could not load demo data: ${esc(e.message)}. Serve this folder over HTTP (e.g. <code>python -m http.server</code>) instead of opening the file directly.</p>`); });

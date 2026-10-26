/* ngram-smoothing: a bigram language model (Markov assumption: each word depends only on the one before it)
 * trained on a tiny corpus, with add-k smoothing
 *     P(w | v) = (c(v, w) + k) / (c(v) + k V),
 * where V counts the possible next tokens (the training words, </s> and <unk>). Words never seen in training
 * become <unk>. Perplexity = exp(-(1/T) Σ log P(x_t | x_{t-1})) over the T predicted tokens of a sentence.
 * At k = 0 an unseen bigram gets probability 0 and the perplexity is infinite; as k grows every P(w | v) tends to
 * 1/V, so the perplexity of any text tends to V.
 * params: {k: 0.1, test: "the dog saw the cat"} */
NB.register("ngram-smoothing", function (el, P) {
  const CORPUS = ["the cat sat on the mat", "the dog sat on the log", "the cat saw a dog", "a dog sat on a mat", "the dog ran"];
  const tok = (s) => s.toLowerCase().split(/[^a-z']+/).filter(Boolean);
  const vocab = [...new Set(CORPUS.flatMap(tok))], V = vocab.length + 2;   // + </s> and <unk>
  const big = {}, uni = {};
  CORPUS.forEach((s) => { const w = ["<s>", ...tok(s), "</s>"];
    for (let i = 1; i < w.length; i++) { const key = w[i - 1] + " " + w[i]; big[key] = (big[key] || 0) + 1; uni[w[i - 1]] = (uni[w[i - 1]] || 0) + 1; } });
  const prob = (v, w, k) => { const c = big[v + " " + w] || 0, cv = uni[v] || 0; return cv + k * V > 0 ? (c + k) / (cv + k * V) : 0; };
  const seq = (s) => ["<s>", ...tok(s).map((w) => (vocab.includes(w) ? w : "<unk>")), "</s>"];
  const ppl = (sents, k) => { let lp = 0, T = 0;
    for (const s of sents) { const w = seq(s); for (let i = 1; i < w.length; i++) { lp += Math.log(prob(w[i - 1], w[i], k)); T++; } }
    return Math.exp(-lp / T); };
  const st = { s: Math.log10(P.k ?? 0.1), test: P.test || "the dog saw the cat" };
  const kOf = (s) => (s <= -3.45 ? 0 : Math.pow(10, s));

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "smoothing k", min: -3.5, max: 2, step: 0.05, value: st.s, format: (v) => (kOf(v) === 0 ? "0" : NB.fmt(kOf(v), kOf(v) < 0.1 ? 3 : 2)) }, (v) => { st.s = v; draw(); });
  const lab = NB.el("label", { class: "nb-select" }, ctl); NB.el("span", { class: "nb-slider-label" }, lab, "test sentence");
  const inp = NB.el("input", { type: "text", value: st.test, size: 22, style: { font: "inherit", padding: "2px 6px", color: "var(--nb-fg)", background: "var(--nb-bg)", border: "1px solid var(--nb-border)", borderRadius: "4px", maxWidth: "100%" } }, lab);
  inp.addEventListener("input", () => { st.test = inp.value; draw(); });
  NB.el("div", { class: "nb-readout", style: { marginTop: 0 } }, el).innerHTML = "training corpus: " + CORPUS.map((s) => `“${s}”`).join(" · ") + ` &nbsp;(V = ${V})`;
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "14px" } }, el);
  const W = 420, H = 290, s1 = NB.svg(row, W, H), s2 = NB.svg(row, W, H);
  NB.legend(el, [["P(word | previous word)", NB.c.blue], ["unseen bigram", NB.c.pink], ["perplexity of the test sentence", NB.c.orange], ["perplexity of the training corpus", NB.c.purple], ["V (uniform model)", NB.c.muted, true]]);
  const read = NB.readout(el);
  const m = { l: 46, r: 10, t: 14, b: 62 };
  const K = NB.scale(-3, 2, m.l, W - m.r), lo = 0, hi = 2;   // log10 axes
  const Q = NB.scale(lo, hi, H - m.b + 24, m.t);

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const k = kOf(st.s), w = seq(st.test), n = w.length - 1;
    // left: per-token probabilities
    const Y = NB.scale(0, 1, H - m.b, m.t), X = NB.scale(0, Math.max(n, 1), m.l, W - m.r), bw = (X(1) - X(0)) * 0.7;
    NB.axes(s1, NB.scale(0, 1, m.l, W - m.r), Y, { yticks: 4, ylabel: "probability", xfmt: () => "", xticks: 1 });
    let lp = 0;
    for (let i = 1; i <= n; i++) {
      const p = prob(w[i - 1], w[i], k), seen = (big[w[i - 1] + " " + w[i]] || 0) > 0, cx = X(i - 0.5);
      lp += Math.log(p);
      NB.el("rect", { x: cx - bw / 2, y: Y(p), width: bw, height: Math.max(0, Y(0) - Y(p)), fill: seen ? NB.c.blue : NB.c.pink }, s1);
      if (!seen) NB.el("rect", { x: cx - bw / 2, y: Y(0) - 3, width: bw, height: 3, fill: NB.c.pink }, s1);
      NB.el("text", { x: cx, y: Y(p) - 4, "text-anchor": "middle", class: "nb-tick" }, s1, p === 0 ? "0" : p.toFixed(2));
      NB.el("text", { x: cx, y: H - m.b + 6, "text-anchor": "end", class: "nb-tick", transform: `rotate(-40 ${cx} ${H - m.b + 6})` }, s1, w[i]);
    }
    const pt = Math.exp(-lp / n);
    // right: perplexity vs k
    NB.axes(s2, K, Q, { xlabel: "smoothing k (log scale)", ylabel: "perplexity", xticks: 5, yticks: 2, xfmt: (t) => String(+Math.pow(10, t).toPrecision(1)), yfmt: (t) => NB.fmt(Math.pow(10, t), 0) });
    NB.el("line", { x1: K(-3), x2: K(2), y1: Q(Math.log10(V)), y2: Q(Math.log10(V)), stroke: NB.c.muted, "stroke-dasharray": "5 4", "stroke-width": 1.4 }, s2);
    const ss = NB.range(-3, 2, 121), cl = (v) => Math.max(lo, Math.min(hi, Math.log10(v)));
    NB.el("path", { d: NB.line(ss.map((s) => [s, cl(ppl(CORPUS, Math.pow(10, s)))]), K, Q), fill: "none", stroke: NB.c.purple, "stroke-width": 2 }, s2);
    NB.el("path", { d: NB.line(ss.map((s) => [s, cl(ppl([st.test], Math.pow(10, s)))]), K, Q), fill: "none", stroke: NB.c.orange, "stroke-width": 2.2 }, s2);
    const kx = k === 0 ? -3 : Math.max(-3, Math.log10(k));
    NB.el("line", { x1: K(kx), x2: K(kx), y1: Q(lo), y2: Q(hi), stroke: NB.c.fg, "stroke-dasharray": "3 3" }, s2);
    NB.el("circle", { cx: K(kx), cy: Q(cl(pt)), r: 5, fill: NB.c.orange }, s2);
    const best = ss.reduce((b, s) => (ppl([st.test], Math.pow(10, s)) < ppl([st.test], Math.pow(10, b)) ? s : b), ss[0]);
    const unseen = w.slice(1).filter((x, i) => !big[w[i] + " " + x]).length, unk = w.filter((x) => x === "<unk>").length;
    read.innerHTML = `k = ${k === 0 ? "0" : NB.fmt(k, 3)}: test perplexity <b>${isFinite(pt) ? NB.fmt(pt, 2) : "∞"}</b> over T = ${n} tokens · training perplexity <b>${NB.fmt(ppl(CORPUS, k), 2)}</b> · ` +
      `${unseen} unseen bigram${unseen === 1 ? "" : "s"}${unk ? `, ${unk} unknown word${unk === 1 ? "" : "s"} (&lt;unk&gt;)` : ""} · test perplexity is lowest ${ppl([st.test], 0) <= ppl([st.test], Math.pow(10, best)) ? "at k = 0" : "near k = " + NB.fmt(Math.pow(10, best), 2)}`;
  }
  draw();
});

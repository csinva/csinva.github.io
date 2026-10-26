/* attention-scores: the T x T attention matrix softmax(Q K^T / sqrt(d)) for a 5-token input, with random
 * query and key vectors (entries ~ N(0, 1), seeded). Each row is one query token and sums to 1.
 * Controls: head dimension d (the first d coordinates of fixed 256-dim vectors), whether to divide the
 * scores by sqrt(d), and the causal (look-ahead) mask.
 * params: {d: 16, scale: true, mask: true} */
NB.register("attention-scores", function (el, P) {
  const toks = ["The", "next", "token", "will", "be"], T = toks.length, DMAX = 256;
  const u = NB.rng(7);
  const Q = toks.map(() => Array.from({ length: DMAX }, () => u.normal()));
  const K = toks.map(() => Array.from({ length: DMAX }, () => u.normal()));
  const dims = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const st = { di: Math.max(0, dims.indexOf(P.d || 16)), scale: P.scale !== false, mask: P.mask !== false };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "head dimension d", min: 0, max: dims.length - 1, step: 1, value: st.di, format: (v) => String(dims[v]) }, (v) => { st.di = v; draw(); });
  NB.select(ctl, "scores", [["yes", "Q·K / √d"], ["no", "Q·K (unscaled)"]], st.scale ? "yes" : "no", (v) => { st.scale = v === "yes"; draw(); });
  NB.select(ctl, "mask", [["yes", "causal (decoder)"], ["no", "none (encoder)"]], st.mask ? "yes" : "no", (v) => { st.mask = v === "yes"; draw(); });
  const cs = 62, L = 92, Tp = 70, svg = NB.svg(el, L + T * cs + 20, Tp + T * cs + 10);
  Object.assign(svg.style, { maxWidth: "480px", display: "block", margin: "0 auto" });
  const read = NB.readout(el);
  function draw() {
    NB.clear(svg);
    const d = dims[st.di];
    const W = Q.map((q, i) => {
      const s = K.map((k, j) => {
        if (st.mask && j > i) return -Infinity;
        let v = 0; for (let c = 0; c < d; c++) v += q[c] * k[c];
        return st.scale ? v / Math.sqrt(d) : v;
      });
      const mx = Math.max(...s), e = s.map((v) => Math.exp(v - mx)), z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / z);
    });
    NB.el("text", { x: L + (T * cs) / 2, y: 16, "text-anchor": "middle", class: "nb-label" }, svg, "keys (attended to)");
    NB.el("text", { x: 12, y: Tp + (T * cs) / 2, "text-anchor": "middle", class: "nb-label", transform: `rotate(-90 12 ${Tp + (T * cs) / 2})` }, svg, "queries");
    toks.forEach((t, j) => NB.el("text", { x: L + j * cs + cs / 2, y: Tp - 12, "text-anchor": "middle", class: "nb-tick", "font-size": 14 }, svg, t));
    toks.forEach((t, i) => NB.el("text", { x: L - 10, y: Tp + i * cs + cs / 2 + 5, "text-anchor": "end", class: "nb-tick", "font-size": 14 }, svg, t));
    W.forEach((row, i) => row.forEach((w, j) => {
      const masked = st.mask && j > i;
      NB.el("rect", { x: L + j * cs + 1, y: Tp + i * cs + 1, width: cs - 2, height: cs - 2, fill: masked ? NB.c.grid : NB.c.blue, "fill-opacity": masked ? 0.5 : 0.06 + 0.9 * w, stroke: NB.c.grid }, svg);
      NB.el("text", { x: L + j * cs + cs / 2, y: Tp + i * cs + cs / 2 + 5, "text-anchor": "middle", style: { fill: w > 0.55 ? NB.c.bg : NB.c.fg }, "font-size": 14 }, svg, masked ? "" : w.toFixed(2));
    }));
    const rows = st.mask ? W.slice(1) : W; // with the mask, the first row is always [1, 0, ...]
    const avgMax = rows.reduce((a, r) => a + Math.max(...r), 0) / rows.length;
    read.innerHTML = `d = ${d}: scores have standard deviation about <b>${st.scale ? "1" : NB.fmt(Math.sqrt(d), 1)}</b>; ` +
      `average largest weight per row <b>${NB.fmt(avgMax, 2)}</b>${st.mask ? " (excluding the first row, which can only see itself)" : ""}. Each row sums to 1.`;
  }
  draw();
});

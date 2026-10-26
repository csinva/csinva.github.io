/* infonce-temperature: softmax weights inside the InfoNCE loss for one query, one positive and N negatives.
 * Low temperature puts almost all weight (and gradient) on the hardest negatives; high temperature
 * spreads it evenly. Colors follow the note's equation: positive pair green (teal there), negatives
 * (the rest of the normalizer) purple, temperature orange.
 * params: {tau: 0.05, negatives: 32} */
NB.register("infonce-temperature", function (el, P) {
  const st = { lt: Math.log10(P.tau || 0.05), n: P.negatives || 32 };
  const sPos = 0.8;
  // fixed pool of negatives: four hard ones (similar to the query) and many easy ones
  const u = NB.rng(7), hard = [0.74, 0.68, 0.62, 0.55];
  const pool = hard.concat(Array.from({ length: 124 }, () => Math.max(-0.3, Math.min(0.45, 0.12 + 0.13 * u.normal()))));
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "temperature τ", min: -2, max: 0, step: 0.02, value: st.lt, format: (v) => NB.fmt(Math.pow(10, v), 3) }, (v) => { st.lt = v; draw(); });
  NB.slider(ctl, { label: "negatives N", min: 4, max: 128, step: 4, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  const W = 640, H = 270, m = { l: 52, r: 10, t: 12, b: 36 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["positive pair (similarity 0.8)", NB.c.green], ["negatives: 4 hard ones (0.55 to 0.74), the rest easy", NB.c.purple]]);
  const read = NB.readout(el);
  function draw() {
    NB.clear(svg);
    const tau = Math.pow(10, st.lt);
    const negs = pool.slice(0, st.n).sort((a, b) => b - a);
    const sims = [sPos].concat(negs);
    const mx = Math.max(...sims) / tau;
    const ex = sims.map((s) => Math.exp(s / tau - mx)), Z = ex.reduce((a, b) => a + b, 0);
    const p = ex.map((e) => e / Z);
    const X = NB.scale(0, sims.length, m.l, W - m.r), Y = NB.scale(0, Math.max(...p) * 1.1, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "candidates, sorted by similarity to the query", ylabel: "softmax weight", xticks: 6, yticks: 4, xfmt: (t) => (t === 0 ? "" : t.toFixed(0)) });
    const bw = X(1) - X(0);
    p.forEach((w, i) => {
      const r = NB.el("rect", { x: X(i) + bw * 0.1, y: Y(w), width: Math.max(0.8, bw * 0.8), height: Y(0) - Y(w), fill: i === 0 ? NB.c.green : NB.c.purple, "fill-opacity": i === 0 ? 0.9 : 0.7 }, svg);
      NB.el("title", {}, r, `${i === 0 ? "positive" : "negative"}: similarity ${NB.fmt(sims[i], 2)}, weight ${NB.fmt(w, 4)}`);
    });
    const negW = p.slice(1).reduce((a, b) => a + b, 0), hardW = p.slice(1, 5).reduce((a, b) => a + b, 0);
    read.innerHTML = `<span style="color:${NB.c.orange}">τ = <b style="color:inherit">${NB.fmt(tau, 3)}</b></span>: loss −log p(positive) = <b>${NB.fmt(-Math.log(p[0]), 3)}</b>. ` +
      `The 4 hard negatives get <b>${NB.fmt((100 * hardW) / negW, 1)}%</b> of the weight that goes to negatives (an even split would give ${NB.fmt((100 * 4) / st.n, 1)}%).`;
  }
  draw();
});

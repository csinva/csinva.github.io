/* ipw-confounding: a confounder X raises both the chance of treatment, e(X) = sigmoid(γX), and the outcome,
 * Y = τT + 2X + ε with τ = 1. Left: one simulated dataset (treated blue, control orange). Right: the estimates of τ from
 * 300 such datasets, for the naive difference in means, the inverse propensity weighting (Horvitz-Thompson) estimator
 * and the Hajek estimator, both using the true e(X).
 * params: {gamma: 1.5, n: 200, reps: 300} */
NB.register("ipw-confounding", function (el, P) {
  const n = P.n || 200, reps = P.reps || 300, tau = 1, bx = 2;
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const sim = (g, u) => Array.from({ length: n }, () => {
    const x = u.normal(), e = sig(g * x), t = u() < e ? 1 : 0;
    return { x, e, t, y: tau * t + bx * x + u.normal() };
  });
  const est = (d) => {
    let s1 = 0, n1 = 0, s0 = 0, n0 = 0, h1 = 0, h0 = 0, w1 = 0, w0 = 0;
    d.forEach(({ e, t, y }) => {
      if (t) { s1 += y; n1++; h1 += y / e; w1 += 1 / e; } else { s0 += y; n0++; h0 += y / (1 - e); w0 += 1 / (1 - e); }
    });
    return { naive: s1 / n1 - s0 / n0, ht: h1 / n - h0 / n, hajek: h1 / w1 - h0 / w0 };
  };
  const st = { g: P.gamma ?? 1.5 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "confounding strength γ", min: 0, max: 4, step: 0.1, value: st.g }, (v) => { st.g = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 420, H = 290, m = { l: 40, r: 10, t: 10, b: 34 }, m2 = { l: 92, r: 12, t: 10, b: 34 };
  const svgL = NB.svg(row, W, H), svgR = NB.svg(row, W, H);
  NB.legend(el, [["treated, T = 1", NB.c.blue], ["control, T = 0", NB.c.orange], ["true effect τ = 1", NB.c.fg, true]]);
  const read = NB.readout(el);
  const rows = [["naive", "difference in means"], ["ht", "IPW (Horvitz-Thompson)"], ["hajek", "Hajek"]];
  function draw() {
    NB.clear(svgL); NB.clear(svgR);
    const one = sim(st.g, NB.rng(101));
    const X = NB.scale(-3, 3, m.l, W - m.r), Y = NB.scale(-7, 8, H - m.b, m.t);
    NB.axes(svgL, X, Y, { xlabel: "confounder X", ylabel: "outcome Y", xticks: 6, yticks: 5 });
    one.forEach(({ x, t, y }) => NB.el("circle", { cx: X(Math.max(-3, Math.min(3, x))), cy: Y(Math.max(-7, Math.min(8, y))), r: 2.6, fill: t ? NB.c.blue : NB.c.orange, "fill-opacity": 0.65 }, svgL));
    const u = NB.rng(7 + Math.round(st.g * 10)), all = Array.from({ length: reps }, () => est(sim(st.g, u)));
    const Xe = NB.scale(-1, 5, m2.l, W - m2.r), rowY = (k) => m.t + 30 + k * 78;
    NB.axes(svgR, Xe, NB.scale(0, 1, H - m.b, m.t), { xlabel: "estimate of τ", xticks: 6, yticks: 1, yfmt: () => "" });
    NB.el("line", { x1: Xe(tau), x2: Xe(tau), y1: m.t, y2: H - m.b, stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.4 }, svgR);
    const u2 = NB.rng(9), stats = {};
    rows.forEach(([k, name], i) => {
      const vals = all.map((a) => a[k]), mean = vals.reduce((s, v) => s + v, 0) / reps, sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / reps);
      stats[k] = { mean, sd };
      const cy = rowY(i) + 14;
      NB.el("text", { x: m2.l - 8, y: cy + 4, class: "nb-tick", "text-anchor": "end" }, svgR, k === "naive" ? "naive" : k === "ht" ? "IPW (HT)" : "Hajek");
      vals.forEach((v) => NB.el("circle", { cx: Xe(Math.max(-1, Math.min(5, v))), cy: cy + (u2() - 0.5) * 34, r: 1.8, fill: NB.c.purple, "fill-opacity": 0.35 }, svgR));
      NB.el("line", { x1: Xe(Math.max(-1, Math.min(5, mean))), x2: Xe(Math.max(-1, Math.min(5, mean))), y1: cy - 22, y2: cy + 22, stroke: NB.c.pink, "stroke-width": 2.5 }, svgR);
    });
    const ex = one.filter((d) => d.e < 0.02 || d.e > 0.98).length;
    read.innerHTML = rows.map(([k, name]) => `${name}: mean <b>${NB.fmt(stats[k].mean, 2)}</b>, sd <b>${NB.fmt(stats[k].sd, 2)}</b>`).join(" · ") +
      ` (pink ticks mark the means). In the dataset on the left, ${ex} of ${n} units have e(X) below 0.02 or above 0.98.`;
  }
  draw();
});

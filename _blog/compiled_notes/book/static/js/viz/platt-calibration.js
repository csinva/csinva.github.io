/* platt-calibration: reliability diagram for an overconfident binary classifier, before and after Platt scaling.
 * True log-odds z ~ N(0, 1.5^2), label y ~ Bernoulli(sigmoid(z)). The classifier outputs the score s = 2.5 z + 0.8,
 * so its probabilities sigmoid(s) are too extreme and shifted. Platt scaling maps s -> sigmoid(A s + B), fit by
 * logistic regression of y on s on a separate calibration set (Newton's method); the perfect map is A = 0.4,
 * B = -0.32. The diagram bins test predictions into 10 equal-width bins; ECE = sum_b (n_b / n) |mean(y) - mean(p)|.
 * Checked against numpy (same rng ported to python) + sklearn LogisticRegression(C=inf) for the fitted A, B and ECE.
 * params: {n: 1500} */
NB.register("platt-calibration", function (el, P) {
  const n = P.n || 1500, sig = (v) => 1 / (1 + Math.exp(-v));
  const u = NB.rng(2024);
  for (let i = 0; i < 5; i++) u();
  const draw2 = () => Array.from({ length: n }, () => { const z = 1.5 * u.normal(); return [2.5 * z + 0.8, u() < sig(z) ? 1 : 0]; });
  const cal = draw2(), test = draw2();
  // Platt scaling: logistic regression of y on [s, 1] by Newton's method
  function fit(data) {
    let A = 0, B = 0;
    for (let it = 0; it < 50; it++) {
      let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
      data.forEach(([s, yy]) => { const p = sig(A * s + B), w = p * (1 - p); g0 += (p - yy) * s; g1 += p - yy; h00 += w * s * s; h01 += w * s; h11 += w; });
      const det = h00 * h11 - h01 * h01, dA = (h11 * g0 - h01 * g1) / det, dB = (h00 * g1 - h01 * g0) / det;
      A -= dA; B -= dB;
      if (Math.abs(dA) + Math.abs(dB) < 1e-10) break;
    }
    return [A, B];
  }
  const st = { A: 1, B: 0 };
  const ctl = NB.controls(el);
  const sA = NB.slider(ctl, { label: "slope A", min: 0.1, max: 1.5, step: 0.01, value: 1 }, (v) => { st.A = v; draw(); });
  const sB = NB.slider(ctl, { label: "intercept B", min: -1.5, max: 1.5, step: 0.01, value: 0 }, (v) => { st.B = v; draw(); });
  NB.button(ctl, "fit Platt scaling", () => { const [A, B] = fit(cal); st.A = A; st.B = B; sA.set(A); sB.set(B); draw(); });
  NB.button(ctl, "raw scores", () => { st.A = 1; st.B = 0; sA.set(1); sB.set(0); draw(); });

  const W = 640, H = 330, m = { l: 50, r: 14, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["observed frequency of y = 1 per bin", NB.c.blue], ["gap to the diagonal", NB.c.pink], ["share of test points per bin", NB.c.muted]]);
  const read = NB.readout(el);
  const pw = 300, X = NB.scale(0, 1, m.l, m.l + pw), Y = NB.scale(0, 1, H - m.b, m.t);
  const X2 = NB.scale(0, 1, 400, W - m.r), Y2 = NB.scale(0, 0.5, H - m.b, m.t);

  function draw() {
    NB.clear(svg);
    const ps = test.map(([s]) => sig(st.A * s + st.B));
    const B = 10, cnt = new Array(B).fill(0), sp = new Array(B).fill(0), sy = new Array(B).fill(0);
    ps.forEach((p, i) => { const b = Math.min(B - 1, Math.floor(p * B)); cnt[b]++; sp[b] += p; sy[b] += test[i][1]; });
    NB.axes(svg, X, Y, { xticks: 5, yticks: 5, xlabel: "predicted probability", ylabel: "observed frequency" });
    NB.el("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: NB.c.muted, "stroke-dasharray": "4 4" }, svg);
    let ece = 0;
    for (let b = 0; b < B; b++) {
      if (!cnt[b]) continue;
      const mp = sp[b] / cnt[b], fy = sy[b] / cnt[b];
      ece += (cnt[b] / n) * Math.abs(fy - mp);
      const x0 = X(b / B) + 2, w = X(1 / B) - X(0) - 4;
      const r = NB.el("rect", { x: x0, y: Y(fy), width: w, height: Y(0) - Y(fy), fill: NB.c.blue, "fill-opacity": 0.75 }, svg);
      NB.el("title", {}, r, `bin ${b / B}-${(b + 1) / B}: ${cnt[b]} points, mean prediction ${NB.fmt(mp, 3)}, observed ${NB.fmt(fy, 3)}`);
      // the gap between prediction and observation: filled above the bar (overconfident), outlined inside it
      NB.el("rect", { x: x0, y: Y(Math.max(fy, mp)), width: w, height: Math.abs(Y(fy) - Y(mp)), fill: fy < mp ? NB.c.pink : "none",
        "fill-opacity": 0.35, stroke: fy < mp ? "none" : NB.c.pink, "stroke-width": 1.5, "stroke-dasharray": "3 2" }, svg);
      NB.el("line", { x1: x0, x2: x0 + w, y1: Y(mp), y2: Y(mp), stroke: NB.c.pink, "stroke-width": 1.5 }, svg);
    }
    // histogram of predictions
    NB.axes(svg, X2, Y2, { xticks: 5, yticks: 5, xlabel: "predicted probability", ylabel: "share of points" });
    for (let b = 0; b < B; b++) {
      const h = cnt[b] / n;
      NB.el("rect", { x: X2(b / B) + 2, y: Y2(Math.min(h, 0.5)), width: X2(1 / B) - X2(0) - 4, height: Y2(0) - Y2(Math.min(h, 0.5)), fill: NB.c.muted, "fill-opacity": 0.6 }, svg);
    }
    const ll = -test.reduce((s, [sc, yy], i) => s + (yy ? Math.log(ps[i]) : Math.log(1 - ps[i])), 0) / n;
    read.innerHTML = `A = ${NB.fmt(st.A, 3)}, B = ${NB.fmt(st.B, 3)}: expected calibration error <b>${NB.fmt(ece, 3)}</b>, ` +
      `test log loss <b>${NB.fmt(ll, 3)}</b>. A &gt; 0 keeps the order of the scores, so the ranking (and the AUC) does not change.`;
  }
  draw();
});

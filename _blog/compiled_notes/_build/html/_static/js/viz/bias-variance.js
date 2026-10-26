/* bias-variance: fit polynomials of a chosen degree to many noisy samples of the same curve.
 * Left: the individual fits, their average, and the truth. Right: bias^2, variance and test error vs degree.
 * params: {n: 20, noise: 0.3, degree: 3, reps: 60} */
NB.register("bias-variance", function (el, P) {
  const n = P.n || 20, noise = P.noise || 0.3, reps = P.reps || 60, maxDeg = 9;   // beyond ~9 plain JS least squares loses precision
  const truth = (x) => Math.sin(2 * Math.PI * x);
  // Legendre polynomials on [-1, 1]: same fits as plain powers of x, but the least-squares system stays well conditioned
  const feats = (x, d) => { const t = 2 * x - 1, L = [1, t]; for (let k = 2; k <= d; k++) L.push(((2 * k - 1) * t * L[k - 1] - (k - 1) * L[k - 2]) / k); return L.slice(0, d + 1); };
  const grid = NB.range(0, 1, 101);
  const rng = NB.rng(7);
  const datasets = Array.from({ length: reps }, () => {
    // one x per interval (a jittered grid): random, but with no big gaps for high degrees to blow up in
    const xs = Array.from({ length: n }, (_, i) => (i + rng()) / n); return { xs, ys: xs.map((x) => truth(x) + noise * rng.normal()) };
  });
  const fitsFor = (d) => datasets.map(({ xs, ys }) => { const w = NB.lstsq(xs.map((x) => feats(x, d)), ys, 1e-3); return grid.map((x) => feats(x, d).reduce((s, f, k) => s + f * w[k], 0)); });
  // bias^2 / variance / expected test error for every degree (computed once)
  const curve = Array.from({ length: maxDeg + 1 }, (_, d) => {
    const F = fitsFor(d), mean = grid.map((_, i) => F.reduce((s, f) => s + f[i], 0) / reps);
    const bias2 = grid.reduce((s, x, i) => s + (mean[i] - truth(x)) ** 2, 0) / grid.length;
    const vr = grid.reduce((s, _, i) => s + F.reduce((t, f) => t + (f[i] - mean[i]) ** 2, 0) / reps, 0) / grid.length;
    return { d, bias2, variance: vr, err: bias2 + vr + noise * noise };
  });

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "polynomial degree", min: 0, max: maxDeg, step: 1, value: P.degree || 3, format: (v) => v.toFixed(0) }, (v) => draw(v));
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 440, H = 300, m = { l: 40, r: 10, t: 10, b: 34 };
  const svgL = NB.svg(row, W, H), svgR = NB.svg(row, W, H);
  NB.legend(el, [["fits to resampled data", NB.c.blue], ["average fit", NB.c.orange], ["true curve", NB.c.fg, true],
                 ["bias²", NB.c.orange], ["variance", NB.c.blue], ["test error", NB.c.pink]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(-2, 2, H - m.b, m.t);
  const Xd = NB.scale(0, maxDeg, m.l, W - m.r), ymax = 0.7, Yd = NB.scale(0, ymax, H - m.b, m.t);

  function draw(d) {
    NB.clear(svgL); NB.clear(svgR);
    NB.axes(svgL, X, Y, { xlabel: "x", ylabel: "y", xticks: 5, yticks: 4 });
    const F = fitsFor(d), mean = grid.map((_, i) => F.reduce((s, f) => s + f[i], 0) / reps);
    F.forEach((f) => NB.el("path", { d: NB.line(grid.map((x, i) => [x, Math.max(-2.2, Math.min(2.2, f[i]))]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-opacity": 0.22, "stroke-width": 1 }, svgL));
    NB.el("path", { d: NB.line(grid.map((x) => [x, truth(x)]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.6 }, svgL);
    NB.el("path", { d: NB.line(grid.map((x, i) => [x, Math.max(-2.2, Math.min(2.2, mean[i]))]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2.4 }, svgL);
    datasets[0].xs.forEach((x, i) => NB.el("circle", { cx: X(x), cy: Y(datasets[0].ys[i]), r: 2.8, fill: NB.c.fg, "fill-opacity": 0.55 }, svgL));

    NB.axes(svgR, Xd, Yd, { xlabel: "polynomial degree (model complexity)", ylabel: "error", xticks: 6, yticks: 4 });
    const clip = (v) => Math.min(v, ymax);
    [["bias2", NB.c.orange], ["variance", NB.c.blue], ["err", NB.c.pink]].forEach(([k, c]) =>
      NB.el("path", { d: NB.line(curve.map((r) => [r.d, clip(r[k])]), Xd, Yd), fill: "none", stroke: c, "stroke-width": 2 }, svgR));
    NB.el("line", { x1: Xd(d), x2: Xd(d), y1: Yd(0), y2: Yd(ymax), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, svgR);
    const r = curve[d], best = curve.reduce((a, b) => (b.err < a.err ? b : a));
    read.innerHTML = `degree ${d}: bias² <b>${NB.fmt(r.bias2, 3)}</b>, variance <b>${NB.fmt(r.variance, 3)}</b>, test error <b>${NB.fmt(r.err, 3)}</b> (noise floor ${NB.fmt(noise * noise, 2)}; lowest at degree ${best.d})`;
  }
  draw(P.degree || 3);
});

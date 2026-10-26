/* covariate-shift: fit a line to y = sin(x) + noise when train and test x come from different distributions.
 * p_train = N(0.5, 0.8^2), p_test = N(0.5 + shift, 0.4^2); p(y | x) is the same in both. The line is
 * misspecified, so the best line depends on where the test points are. Ordinary least squares fits the
 * training region; importance-weighted least squares with w(x) = p_test(x) / p_train(x) targets the test
 * region, at the cost of an effective sample size (sum w)^2 / sum w^2 that drops as the weights get extreme.
 * Test error = E_test[(fit(x) - sin x)^2] (excess over the noise), integrated on a grid. Checked against
 * numpy (np.linalg.lstsq / weighted normal equations, scipy.stats.norm) on the same data.
 * params: {shift: 1.5, n: 80, noise: 0.25, seed: 11} */
NB.register("covariate-shift", function (el, P) {
  const n = P.n || 80, noise = P.noise === undefined ? 0.25 : P.noise, mTr = 0.5, sTr = 0.8, sTe = 0.4;
  const st = { shift: P.shift === undefined ? 1.5 : P.shift };
  const f = Math.sin;
  const pdf = (x, mu, s) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const rng = NB.rng(P.seed || 11);
  for (let i = 0; i < 20; i++) rng();
  let xs = [], ys = [];
  const sample = () => { xs = Array.from({ length: n }, () => mTr + sTr * rng.normal()); ys = xs.map((x) => f(x) + noise * rng.normal()); };
  const fit = (w, X = xs, Yv = ys) => NB.lstsq(X.map((x, i) => [Math.sqrt(w[i]), Math.sqrt(w[i]) * x]), Yv.map((y, i) => Math.sqrt(w[i]) * y));
  // 200 other training samples (fixed), to report the average test error of each method at the current shift
  const rngB = NB.rng(48271);
  for (let i = 0; i < 20; i++) rngB();
  const reps = Array.from({ length: 200 }, () => { const X = Array.from({ length: n }, () => mTr + sTr * rngB.normal()); return { X, Y: X.map((x) => f(x) + noise * rngB.normal()) }; });
  const fmt3 = (v) => (v < 0.001 ? "<0.001" : v.toFixed(3));
  const testErr = (b, mu) => {
    const g = NB.range(mu - 4 * sTe, mu + 4 * sTe, 401); let s = 0, z = 0;
    g.forEach((x) => { const p = pdf(x, mu, sTe); s += p * (b[0] + b[1] * x - f(x)) ** 2; z += p; });
    return s / z;
  };

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "shift of p_test", min: 0, max: 2.5, step: 0.05, value: st.shift }, (v) => { st.shift = v; draw(); });
  NB.button(ctl, "new training sample", () => { sample(); draw(); });

  const W = 640, H = 380, m = { l: 44, r: 14, t: 12 }, yb = 250, stripTop = 290, stripBot = H - 40;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["true curve sin x", NB.c.fg, true], ["least squares", NB.c.orange], ["importance-weighted", NB.c.blue],
                 ["p_train", NB.c.green], ["p_test", NB.c.pink]]);
  const read = NB.readout(el);
  const X = NB.scale(-2, 4.5, m.l, W - m.r), Y = NB.scale(-1.6, 2, yb, m.t);
  const D = NB.scale(0, 1.05, stripBot, stripTop);
  const uid = "cs" + Math.random().toString(36).slice(2, 8);

  function draw() {
    NB.clear(svg);
    const mu = mTr + st.shift;
    const w = xs.map((x) => pdf(x, mu, sTe) / pdf(x, mTr, sTr));
    const b0 = fit(xs.map(() => 1)), b1 = fit(w);
    const e0 = testErr(b0, mu), e1 = testErr(b1, mu);
    const sw = w.reduce((a, b) => a + b, 0), ess = (sw * sw) / w.reduce((a, b) => a + b * b, 0), wmax = Math.max(...w);

    // main panel: data, truth, both fits
    NB.axes(svg, X, Y, { ylabel: "y", xticks: 7, yticks: 4, xfmt: () => "" });
    const cp = NB.el("clipPath", { id: uid }, NB.el("defs", {}, svg));
    NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: yb - m.t }, cp);
    const g = NB.el("g", { "clip-path": `url(#${uid})` }, svg);
    NB.el("rect", { x: X(mu - 2 * sTe), y: m.t, width: X(mu + 2 * sTe) - X(mu - 2 * sTe), height: yb - m.t, fill: NB.c.pink, "fill-opacity": 0.07 }, g);
    xs.forEach((x, i) => NB.el("circle", { cx: X(x), cy: Y(ys[i]), r: 1.6 + 5 * Math.sqrt(w[i] / wmax), fill: NB.c.muted, "fill-opacity": 0.55 }, g));
    const grid = NB.range(-2, 4.5, 200);
    NB.el("path", { d: NB.line(grid.map((x) => [x, f(x)]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, g);
    [[b0, NB.c.orange], [b1, NB.c.blue]].forEach(([b, c]) =>
      NB.el("path", { d: NB.line([[-2, b[0] - 2 * b[1]], [4.5, b[0] + 4.5 * b[1]]], X, Y), fill: "none", stroke: c, "stroke-width": 2.4 }, g));
    NB.el("text", { x: X(mu), y: yb - 8, "text-anchor": "middle", class: "nb-tick", style: { fill: NB.c.pink } }, svg, "test region");

    // strip: the two x densities on the same x axis
    NB.el("line", { x1: m.l, x2: W - m.r, y1: stripBot, y2: stripBot, class: "nb-axis" }, svg);
    NB.ticks(-2, 4.5, 7).forEach((t) => NB.el("text", { x: X(t), y: stripBot + 16, class: "nb-tick", "text-anchor": "middle" }, svg, NB.fmt(t)));
    NB.el("text", { x: (m.l + W - m.r) / 2, y: stripBot + 34, class: "nb-label", "text-anchor": "middle" }, svg, "x");
    NB.el("text", { x: m.l - 6, y: stripTop + 10, class: "nb-tick", "text-anchor": "end" }, svg, "p(x)");
    const top = Math.max(pdf(mTr, mTr, sTr), pdf(mu, mu, sTe));
    [[mTr, sTr, NB.c.green], [mu, sTe, NB.c.pink]].forEach(([c, s, col]) => {
      const pts = grid.map((x) => [x, pdf(x, c, s) / top]);
      NB.el("path", { d: NB.line(pts, X, D) + `L${X(4.5)},${D(0)}L${X(-2)},${D(0)}Z`, fill: col, "fill-opacity": 0.18, stroke: col, "stroke-width": 1.6 }, svg);
    });

    let a0 = 0, a1 = 0;
    reps.forEach(({ X: Xr, Y: Yr }) => {
      const wr = Xr.map((x) => pdf(x, mu, sTe) / pdf(x, mTr, sTr));
      a0 += testErr(fit(Xr.map(() => 1), Xr, Yr), mu) / reps.length; a1 += testErr(fit(wr, Xr, Yr), mu) / reps.length;
    });
    read.innerHTML = `Test error (mean squared distance to sin x under p_test) on this sample: least squares <b>${fmt3(e0)}</b>, ` +
      `importance-weighted <b>${fmt3(e1)}</b>; averaged over 200 training samples: ${fmt3(a0)} vs ${fmt3(a1)}. ` +
      `Effective sample size of the weighted fit: <b>${NB.fmt(ess, 1)}</b> of ${n} points (largest weight ${NB.fmt(wmax, 1)})` +
      (ess < 8 ? ", so a handful of points decide the weighted line and it changes a lot from sample to sample." : ".");
  }
  sample();
  draw();
});

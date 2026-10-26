/* kernel-smoother: Nadaraya-Watson (local constant) or lowess-style local linear smoothing of noisy data
 * with a Gaussian kernel K_h. Left: data, true curve and the smooth, with the kernel window drawn at x = 0.5.
 * Right: leave-one-out cross-validation error against the bandwidth h (log scale), which picks a good h.
 * params: {h: 0.08, n: 60, noise: 0.35, method: "nw" | "local-linear"} */
NB.register("kernel-smoother", function (el, P) {
  const n = P.n || 60, noise = P.noise || 0.35;
  const truth = (x) => Math.sin(2 * Math.PI * x) + 1.5 * x;
  const rng = NB.rng(5);
  const xs = Array.from({ length: n }, () => rng()).sort((a, b) => a - b);
  const ys = xs.map((x) => truth(x) + noise * rng.normal());
  const K = (t, h) => Math.exp(-0.5 * (t / h) ** 2) / h;          // K_h(t) = K(t/h) / h with a Gaussian K (its constant cancels)
  // smoothed value at x0, optionally leaving out point `skip`
  function smooth(x0, h, method, skip) {
    let s0 = 0, s1 = 0, s2 = 0, t0 = 0, t1 = 0;
    for (let i = 0; i < n; i++) {
      if (i === skip) continue;
      const d = xs[i] - x0, w = K(d, h);
      s0 += w; s1 += w * d; s2 += w * d * d; t0 += w * ys[i]; t1 += w * d * ys[i];
    }
    if (!(s0 > 1e-300)) return NaN;
    if (method === "local-linear") {                               // weighted least-squares line around x0, read off at x0
      const det = s0 * s2 - s1 * s1;
      if (det > 1e-9 * s0 * s2) return (s2 * t0 - s1 * t1) / det;
    }
    return t0 / s0;                                                // g_h(x0) = sum K_h(x_i - x0) y_i / sum K_h(x_i - x0)
  }
  const L0 = Math.log10(0.005), L1 = Math.log10(0.5);
  const cvAt = (h, method) => {
    let s = 0, c = 0;
    for (let i = 0; i < n; i++) { const g = smooth(xs[i], h, method, i); if (isFinite(g)) { s += (ys[i] - g) ** 2; c++; } }
    return s / c;
  };
  const cvCurve = {};
  const cv = (method) => cvCurve[method] || (cvCurve[method] = NB.range(L0, L1, 60).map((l) => [l, cvAt(10 ** l, method)]));

  const st = { h: P.h || 0.08, method: P.method || "nw" };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "bandwidth h", min: L0, max: L1, step: 0.01, value: Math.log10(st.h), format: (v) => NB.fmt(10 ** v, 3) }, (v) => { st.h = 10 ** v; draw(); });
  NB.select(ctl, "smoother", [["nw", "Nadaraya-Watson (local average)"], ["local-linear", "local linear (lowess)"]], st.method, (v) => { st.method = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1.4fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 460, H = 300, W2 = 330, m = { l: 40, r: 10, t: 10, b: 34 }, m2 = { l: 48, r: 12, t: 10, b: 34 };
  const svgL = NB.svg(row, W, H), svgR = NB.svg(row, W2, H);
  NB.legend(el, [["data", NB.c.fg], ["true curve", NB.c.fg, true], ["smooth", NB.c.blue], ["kernel weights at x = 0.5", NB.c.orange], ["leave-one-out CV error", NB.c.pink]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(-1.5, 3, H - m.b, m.t);
  const grid = NB.range(0, 1, 201);

  function draw() {
    NB.clear(svgL); NB.clear(svgR);
    NB.axes(svgL, X, Y, { xlabel: "x", ylabel: "y", xticks: 5, yticks: 5 });
    // kernel window at x = 0.5, drawn as a bump on the floor of the plot
    const kmax = K(0, st.h), kh = (x) => Y(-1.5) - 55 * K(x - 0.5, st.h) / kmax;
    NB.el("path", { d: "M" + grid.map((x) => X(x).toFixed(1) + "," + kh(x).toFixed(1)).join("L") + `L${X(1)},${Y(-1.5)}L${X(0)},${Y(-1.5)}Z`, fill: NB.c.orange, "fill-opacity": 0.18, stroke: NB.c.orange, "stroke-width": 1.2 }, svgL);
    NB.el("path", { d: NB.line(grid.map((x) => [x, truth(x)]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.4 }, svgL);
    xs.forEach((x, i) => {
      const w = K(x - 0.5, st.h) / kmax;
      NB.el("circle", { cx: X(x), cy: Y(ys[i]), r: 3, fill: NB.c.fg, "fill-opacity": 0.35 }, svgL);
      if (w > 0.02) NB.el("circle", { cx: X(x), cy: Y(ys[i]), r: 3 + 2.5 * w, fill: "none", stroke: NB.c.orange, "stroke-opacity": 0.3 + 0.7 * w, "stroke-width": 1.5 }, svgL);
    });
    const g = grid.map((x) => [x, Math.max(-1.6, Math.min(3.1, smooth(x, st.h, st.method)))]);
    NB.el("path", { d: NB.line(g, X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.4 }, svgL);

    const c = cv(st.method), cmax = 0.5;
    const Xh = NB.scale(L0, L1, m2.l, W2 - m2.r), Yc = NB.scale(0, cmax, H - m2.b, m2.t);
    NB.axes(svgR, Xh, Yc, { xlabel: "bandwidth h (log scale)", ylabel: "CV error", xticks: 3, yticks: 5, xfmt: (t) => NB.fmt(10 ** t, 3) });
    NB.el("path", { d: NB.line(c.map(([l, e]) => [l, Math.min(e, cmax)]), Xh, Yc), fill: "none", stroke: NB.c.pink, "stroke-width": 2 }, svgR);
    NB.el("line", { x1: Xh(Math.log10(st.h)), x2: Xh(Math.log10(st.h)), y1: Yc(0), y2: Yc(cmax), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, svgR);
    const best = c.reduce((a, b) => (b[1] < a[1] ? b : a));
    NB.el("circle", { cx: Xh(best[0]), cy: Yc(best[1]), r: 4, fill: NB.c.pink }, svgR);
    read.innerHTML = `h = ${NB.fmt(st.h, 3)}: leave-one-out CV error <b>${NB.fmt(cvAt(st.h, st.method), 3)}</b> (noise variance ${NB.fmt(noise * noise, 3)}); lowest CV error at h ≈ <b>${NB.fmt(10 ** best[0], 3)}</b>`;
  }
  draw();
});

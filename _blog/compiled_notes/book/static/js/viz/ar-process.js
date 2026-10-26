/* ar-process: simulate an AR(1) process X_t = φ X_{t-1} + ε_t (white noise ε_t ~ N(0, 1), X_0 = 0), always with the
 * same noise draws. Left: the series. Right: its sample autocorrelation by lag, with φ^k, the autocorrelation of the
 * stationary process. |φ| < 1 gives a stationary series; |φ| ≥ 1 wanders off or blows up.
 * params: {phi: 0.7, n: 200} */
NB.register("ar-process", function (el, P) {
  const n = P.n || 200, rng = NB.rng(17), eps = Array.from({ length: n }, () => rng.normal()), maxLag = 20;
  const st = { phi: P.phi ?? 0.7 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "φ", min: -1.1, max: 1.1, step: 0.01, value: st.phi }, (v) => { st.phi = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1.6fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 500, H = 280, W2 = 310, m = { l: 40, r: 10, t: 10, b: 34 };
  const svgL = NB.svg(row, W, H), svgR = NB.svg(row, W2, H);
  NB.legend(el, [["X_t", NB.c.blue], ["sample autocorrelation", NB.c.orange], ["φ^k, autocorrelation of the stationary AR(1) (open circles)", NB.c.fg]]);
  const read = NB.readout(el);
  const Xt = NB.scale(0, n, m.l, W - m.r), Yx = NB.scale(-10, 10, H - m.b, m.t);
  const Xk = NB.scale(0, maxLag, m.l, W2 - m.r), Yk = NB.scale(-1, 1, H - m.b, m.t);
  const defs = NB.el("defs", {}, svgL);
  const clip = NB.el("clipPath", { id: "ar-clip-" + Math.random().toString(36).slice(2) }, defs);
  NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, clip);
  const gL = NB.el("g", {}, svgL), gL2 = NB.el("g", { "clip-path": `url(#${clip.id})` }, svgL);
  function draw() {
    NB.clear(gL); NB.clear(gL2); NB.clear(svgR);
    const x = [0];
    for (let t = 1; t < n; t++) x.push(st.phi * x[t - 1] + eps[t]);
    NB.axes(gL, Xt, Yx, { xlabel: "time t", ylabel: "X_t", xticks: 5, yticks: 4 });
    NB.el("path", { d: NB.line(x.map((v, t) => [t, Math.max(-1e4, Math.min(1e4, v))]), Xt, Yx), fill: "none", stroke: NB.c.blue, "stroke-width": 1.5 }, gL2);
    const mu = x.reduce((s, v) => s + v, 0) / n, c0 = x.reduce((s, v) => s + (v - mu) ** 2, 0);
    const acf = Array.from({ length: maxLag + 1 }, (_, k) => { let s = 0; for (let t = 0; t + k < n; t++) s += (x[t] - mu) * (x[t + k] - mu); return s / c0; });
    NB.axes(svgR, Xk, Yk, { xlabel: "lag k", ylabel: "autocorrelation", xticks: 4, yticks: 4 });
    NB.el("line", { x1: Xk(0), x2: Xk(maxLag), y1: Yk(0), y2: Yk(0), stroke: NB.c.muted }, svgR);
    acf.forEach((a, k) => {
      NB.el("line", { x1: Xk(k), x2: Xk(k), y1: Yk(0), y2: Yk(a), stroke: NB.c.orange, "stroke-width": 2.5 }, svgR);
      NB.el("circle", { cx: Xk(k), cy: Yk(a), r: 2.5, fill: NB.c.orange }, svgR);
    });
    if (Math.abs(st.phi) < 1) acf.forEach((_, k) => NB.el("circle", { cx: Xk(k), cy: Yk(st.phi ** k), r: 4.5, fill: "none", stroke: NB.c.fg, "stroke-width": 1.2 }, svgR));
    const off = x.some((v) => Math.abs(v) > 10);
    read.innerHTML = Math.abs(st.phi) < 1
      ? `|φ| < 1: stationary. Stationary variance 1/(1 - φ²) = <b>${NB.fmt(1 / (1 - st.phi * st.phi), 2)}</b>, sample variance <b>${NB.fmt(c0 / n, 2)}</b>; autocorrelation at lag 1 is φ = ${NB.fmt(st.phi, 2)}`
      : `|φ| ≥ 1: not stationary. The variance grows with t${off ? " and the series leaves the plot" : ""}, so the sample autocorrelation no longer estimates anything fixed.`;
  }
  draw();
});

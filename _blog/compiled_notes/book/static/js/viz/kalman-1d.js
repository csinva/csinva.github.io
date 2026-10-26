/* kalman-1d: Kalman filter for a 1-d random walk x_{t+1} = x_t + w_t, w ~ N(0, σ_x²), observed as y_t = x_t + v_t,
 * v ~ N(0, σ_z²) (A = C = G = 1). The new mean is a weighted mean of the new observation and the predicted old mean,
 * with weight K_t (the gain) on the observation. The variance update never looks at the observations and settles
 * quickly to a fixed value that depends only on σ_x² and σ_z².
 * params: {sx: 0.5, sz: 1.5, T: 60} */
NB.register("kalman-1d", function (el, P) {
  const T = P.T || 60, rng = NB.rng(23), w = Array.from({ length: T }, () => rng.normal()), v = Array.from({ length: T }, () => rng.normal());
  const st = { sx: P.sx || 0.5, sz: P.sz || 1.5 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "state noise σ_x", min: 0.05, max: 2, step: 0.05, value: st.sx }, (x) => { st.sx = x; draw(); });
  NB.slider(ctl, { label: "sensor noise σ_z", min: 0.1, max: 3, step: 0.05, value: st.sz }, (x) => { st.sz = x; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1.6fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 500, H = 280, W2 = 310, m = { l: 40, r: 10, t: 10, b: 34 }, m2 = { l: 44, r: 10, t: 10, b: 34 };
  const svgL = NB.svg(row, W, H), svgR = NB.svg(row, W2, H);
  NB.legend(el, [["true state x_t", NB.c.fg, true], ["observations y_t", NB.c.orange], ["filtered mean ± 2 sd", NB.c.blue], ["gain K_t", NB.c.purple]]);
  const read = NB.readout(el);
  function draw() {
    NB.clear(svgL); NB.clear(svgR);
    const x = [0], y = [];
    for (let t = 1; t < T; t++) x.push(x[t - 1] + st.sx * w[t]);
    for (let t = 0; t < T; t++) y.push(x[t] + st.sz * v[t]);
    // prior on x_0: mean 0, variance 4
    let mu = 0, Pv = 4; const mus = [], sds = [], Ks = [];
    for (let t = 0; t < T; t++) {
      const Pm = t === 0 ? Pv : Pv + st.sx ** 2;          // predict: same mean, variance grows by σ_x²
      const K = Pm / (Pm + st.sz ** 2);                    // update: weighted mean of prediction and observation
      mu = mu + K * (y[t] - mu); Pv = (1 - K) * Pm;
      mus.push(mu); sds.push(Math.sqrt(Pv)); Ks.push(K);
    }
    const lo = Math.min(...y, ...x) - 0.5, hi = Math.max(...y, ...x) + 0.5;
    const X = NB.scale(0, T - 1, m.l, W - m.r), Y = NB.scale(lo, hi, H - m.b, m.t);
    NB.axes(svgL, X, Y, { xlabel: "time t", ylabel: "position", xticks: 6, yticks: 5 });
    const band = mus.map((u, t) => [t, u + 2 * sds[t]]).concat(mus.map((u, t) => [t, u - 2 * sds[t]]).reverse());
    NB.el("path", { d: NB.line(band, X, Y) + "Z", fill: NB.c.blue, "fill-opacity": 0.15, stroke: "none" }, svgL);
    NB.el("path", { d: NB.line(x.map((u, t) => [t, u]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svgL);
    y.forEach((u, t) => NB.el("circle", { cx: X(t), cy: Y(u), r: 2.6, fill: NB.c.orange, "fill-opacity": 0.8 }, svgL));
    NB.el("path", { d: NB.line(mus.map((u, t) => [t, u]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.2 }, svgL);
    const Xk = NB.scale(0, T - 1, m2.l, W2 - m2.r), Yk = NB.scale(0, 1, H - m2.b, m2.t);
    NB.axes(svgR, Xk, Yk, { xlabel: "time t", ylabel: "gain K_t", xticks: 3, yticks: 4 });
    NB.el("path", { d: NB.line(Ks.map((k, t) => [t, k]), Xk, Yk), fill: "none", stroke: NB.c.purple, "stroke-width": 2.2 }, svgR);
    const q = st.sx ** 2, r = st.sz ** 2, Pss = (-q + Math.sqrt(q * q + 4 * q * r)) / 2, Kss = (Pss + q) / (Pss + q + r);
    const rmseF = Math.sqrt(mus.reduce((s, u, t) => s + (u - x[t]) ** 2, 0) / T), rmseY = Math.sqrt(y.reduce((s, u, t) => s + (u - x[t]) ** 2, 0) / T);
    read.innerHTML = `steady state: gain K = <b>${NB.fmt(Kss, 3)}</b>, filtered sd = <b>${NB.fmt(Math.sqrt(Pss), 3)}</b> · error of the filtered mean <b>${NB.fmt(rmseF, 3)}</b> vs raw observations <b>${NB.fmt(rmseY, 3)}</b> (root mean square)`;
  }
  draw();
});

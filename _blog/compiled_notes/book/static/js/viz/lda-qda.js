/* lda-qda: Gaussian class-conditioned classifier on 2-d data. Each class gets a Gaussian fit by MLE; the
 * covariances are shrunk toward the pooled one, Σ_k = α Σ_k + (1 - α) Σ (regularized discriminant analysis).
 * α = 0 is LDA (shared Σ, straight boundary), α = 1 is QDA (separate Σ_k, quadratic boundary).
 * Shading is the posterior p(Y = 1 | x); the solid curve is where it equals 1/2.
 * params: {alpha: 1, spread: 2.2, tilt: 0.6, n: 80} */
NB.register("lda-qda", function (el, P) {
  const st = { alpha: P.alpha ?? 1, spread: P.spread ?? 2.2, tilt: P.tilt ?? 0.6 }, n = P.n || 80;
  const W = 620, H = 400, m = { l: 36, r: 10, t: 10, b: 30 };
  const X = NB.scale(-5, 5, m.l, W - m.r), Y = NB.scale(-3.2, 3.2, H - m.b, m.t);
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "class 1 spread", min: 0.4, max: 3, step: 0.05, value: st.spread }, (v) => { st.spread = v; sample(); draw(); });
  NB.slider(ctl, { label: "class 1 tilt", min: -0.9, max: 0.9, step: 0.05, value: st.tilt }, (v) => { st.tilt = v; sample(); draw(); });
  NB.slider(ctl, { label: "α (0 = LDA, 1 = QDA)", min: 0, max: 1, step: 0.05, value: st.alpha }, (v) => { st.alpha = v; draw(); });
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["class 0", NB.c.blue], ["class 1", NB.c.orange], ["boundary, p(Y=1 | x) = 1/2", NB.c.fg], ["fitted covariance (2σ)", NB.c.muted, true]]);
  const read = NB.readout(el);

  // the same standard-normal draws every time, so a slider reshapes the cloud instead of resampling it
  const rng = NB.rng(11), Z0 = Array.from({ length: n }, () => [rng.normal(), rng.normal()]), Z1 = Array.from({ length: n }, () => [rng.normal(), rng.normal()]);
  let D0, D1;
  function sample() {
    D0 = Z0.map(([a, b]) => [-1.4 + 0.9 * a, 0.9 * b]);
    const s = st.spread, r = st.tilt;   // class 1: x scale s, correlation r
    D1 = Z1.map(([a, b]) => [1.4 + 0.9 * s * a, 0.9 * (r * a + Math.sqrt(1 - r * r) * b)]);
  }
  const mean = (D) => [D.reduce((t, p) => t + p[0], 0) / D.length, D.reduce((t, p) => t + p[1], 0) / D.length];
  const cov = (D, mu) => { const c = [[0, 0], [0, 0]]; D.forEach((p) => { const d = [p[0] - mu[0], p[1] - mu[1]]; for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) c[i][j] += (d[i] * d[j]) / D.length; }); return c; };
  const inv2 = (c) => { const det = c[0][0] * c[1][1] - c[0][1] * c[1][0]; return { det, inv: [[c[1][1] / det, -c[0][1] / det], [-c[1][0] / det, c[0][0] / det]] }; };
  const mix = (a, b, w) => a.map((r, i) => r.map((v, j) => w * v + (1 - w) * b[i][j]));

  function draw() {
    NB.clear(svg);
    const mu0 = mean(D0), mu1 = mean(D1), C0 = cov(D0, mu0), C1 = cov(D1, mu1);
    const pooled = mix(C0, C1, 0.5);                                   // equal class sizes
    const Cs = [mix(C0, pooled, st.alpha), mix(C1, pooled, st.alpha)], S = Cs.map(inv2), mus = [mu0, mu1];
    const g = (x, y, k) => { const d = [x - mus[k][0], y - mus[k][1]], I = S[k].inv; return -0.5 * Math.log(S[k].det) - 0.5 * (d[0] * (I[0][0] * d[0] + I[0][1] * d[1]) + d[1] * (I[1][0] * d[0] + I[1][1] * d[1])); };
    const f = (x, y) => g(x, y, 1) - g(x, y, 0);                      // log odds (equal priors)
    const [xa, xb] = X.domain, [ya, yb] = Y.domain;
    // posterior shading
    const nx = 62, ny = 40, dx = (xb - xa) / nx, dy = (yb - ya) / ny, sh = NB.el("g", {}, svg);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = xa + (i + 0.5) * dx, y = ya + (j + 0.5) * dy, p = 1 / (1 + Math.exp(-f(x, y)));
      NB.el("rect", { x: X(x - dx / 2), y: Y(y + dy / 2), width: X(dx) - X(0) + 0.6, height: Y(0) - Y(dy) + 0.6,
        fill: p > 0.5 ? NB.c.orange : NB.c.blue, "fill-opacity": (0.3 * Math.abs(p - 0.5) * 2).toFixed(3) }, sh);
    }
    NB.axes(svg, X, Y, { grid: false, xlabel: "x₁", ylabel: "x₂", xticks: 5, yticks: 4 });
    // boundary f = 0 by marching squares
    const gx = 120, gy = 78, V = [];
    for (let j = 0; j <= gy; j++) { V.push([]); for (let i = 0; i <= gx; i++) V[j].push(f(xa + (i / gx) * (xb - xa), ya + (j / gy) * (yb - ya))); }
    const Pt = (i, j) => [X(xa + (i / gx) * (xb - xa)), Y(ya + (j / gy) * (yb - ya))];
    let d = "";
    for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
      const c = [V[j][i], V[j][i + 1], V[j + 1][i + 1], V[j + 1][i]], q = [Pt(i, j), Pt(i + 1, j), Pt(i + 1, j + 1), Pt(i, j + 1)], pts = [];
      for (let e = 0; e < 4; e++) { const a = c[e], b = c[(e + 1) % 4]; if (a * b < 0) { const t = a / (a - b), A = q[e], B = q[(e + 1) % 4]; pts.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); } }
      for (let k = 0; k + 1 < pts.length; k += 2) d += `M${pts[k][0].toFixed(1)},${pts[k][1].toFixed(1)}L${pts[k + 1][0].toFixed(1)},${pts[k + 1][1].toFixed(1)}`;
    }
    NB.el("path", { d, fill: "none", stroke: NB.c.fg, "stroke-width": 2.2 }, svg);
    // the covariances actually used, as 2σ ellipses
    Cs.forEach((C, k) => {
      const a = C[0][0], b = C[0][1], c2 = C[1][1], tr = (a + c2) / 2, disc = Math.sqrt(((a - c2) / 2) ** 2 + b * b);
      const l1 = tr + disc, l2 = Math.max(tr - disc, 0), th = Math.abs(b) < 1e-12 ? (a >= c2 ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
      const pts = NB.range(0, 2 * Math.PI, 80).map((t) => { const u = 2 * Math.sqrt(l1) * Math.cos(t), v = 2 * Math.sqrt(l2) * Math.sin(t); return [mus[k][0] + u * Math.cos(th) - v * Math.sin(th), mus[k][1] + u * Math.sin(th) + v * Math.cos(th)]; });
      NB.el("path", { d: NB.line(pts, X, Y) + "Z", fill: "none", stroke: k ? NB.c.orange : NB.c.blue, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    });
    const cx = (v) => X(Math.max(xa, Math.min(xb, v))), cy = (v) => Y(Math.max(ya, Math.min(yb, v)));
    D0.forEach((p) => NB.el("circle", { cx: cx(p[0]), cy: cy(p[1]), r: 3, fill: NB.c.blue, "fill-opacity": 0.85 }, svg));
    D1.forEach((p) => NB.el("rect", { x: cx(p[0]) - 2.8, y: cy(p[1]) - 2.8, width: 5.6, height: 5.6, fill: NB.c.orange, "fill-opacity": 0.85 }, svg));
    const acc = (D0.filter((p) => f(...p) < 0).length + D1.filter((p) => f(...p) > 0).length) / (2 * n);
    const kind = st.alpha === 0 ? "<b>LDA</b>: one shared covariance, so the boundary is a straight line"
      : st.alpha === 1 ? "<b>QDA</b>: a covariance per class, so the boundary is quadratic" : "between LDA and QDA";
    read.innerHTML = `α = ${NB.fmt(st.alpha, 2)}, ${kind} · training accuracy <b>${(100 * acc).toFixed(1)}%</b>`;
  }
  sample(); draw();
});

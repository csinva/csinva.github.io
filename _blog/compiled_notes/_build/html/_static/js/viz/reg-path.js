/* reg-path: coefficient paths of ridge, argmin ||y - Xβ||² + λ||β||₂², and lasso, argmin ||y - Xβ||² + λ||β||₁,
 * on one simulated dataset with 6 standardized features (x1 and x2 strongly correlated). A slider moves λ along both
 * paths. Ridge shrinks every coefficient smoothly; lasso sets coefficients exactly to 0 one after another.
 * params: {lam: 30, n: 40} */
NB.register("reg-path", function (el, P) {
  const n = P.n || 40, p = 6, beta = [2, 0.6, -1.5, 0, 1, 0];
  const rng = NB.rng(3);
  const raw = Array.from({ length: n }, () => {
    const z = rng.normal(), r = [z, 0.9 * z + Math.sqrt(1 - 0.81) * rng.normal()];
    for (let j = 2; j < p; j++) r.push(rng.normal());
    return r;
  });
  const cols = Array.from({ length: p }, (_, j) => raw.map((r) => r[j]));
  const X = raw.map((r) => r.map((v, j) => {                        // center and scale each column to mean 0, sd 1
    const c = cols[j], mu = c.reduce((s, x) => s + x, 0) / n, sd = Math.sqrt(c.reduce((s, x) => s + (x - mu) ** 2, 0) / n);
    return (v - mu) / sd;
  }));
  let y = X.map((r) => r.reduce((s, v, j) => s + v * beta[j], 0) + 1.5 * rng.normal());
  const ybar = y.reduce((s, v) => s + v, 0) / n; y = y.map((v) => v - ybar);   // centered, so no intercept
  const xx = Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[j] * r[j], 0));
  const lasso = (lam, init) => {                                   // coordinate descent; soft-threshold at λ/2
    const b = init ? init.slice() : new Array(p).fill(0), res = y.map((v, i) => v - X[i].reduce((s, x, j) => s + x * b[j], 0));
    for (let it = 0; it < 500; it++) {
      let delta = 0;
      for (let j = 0; j < p; j++) {
        let rho = 0; for (let i = 0; i < n; i++) rho += X[i][j] * (res[i] + X[i][j] * b[j]);
        const nb = Math.sign(rho) * Math.max(Math.abs(rho) - lam / 2, 0) / xx[j];
        if (nb !== b[j]) { for (let i = 0; i < n; i++) res[i] -= X[i][j] * (nb - b[j]); delta = Math.max(delta, Math.abs(nb - b[j])); b[j] = nb; }
      }
      if (delta < 1e-9) break;
    }
    return b;
  };
  const lamMax = 2 * Math.max(...Array.from({ length: p }, (_, j) => Math.abs(X.reduce((s, r, i) => s + r[j] * y[i], 0))));
  const L0 = Math.log10(lamMax) - 3, L1 = Math.log10(lamMax) + 0.3;
  const ls = NB.range(L0, L1, 80);
  let warm = null;
  const lassoPath = ls.map((l) => (warm = lasso(10 ** l, warm)));
  const ridgePath = ls.map((l) => NB.lstsq(X, y, 10 ** l));

  const st = { l: Math.log10(P.lam || 30) };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "penalty λ", min: L0, max: L1, step: 0.01, value: Math.min(L1, Math.max(L0, st.l)), format: (v) => NB.fmt(10 ** v, 1) }, (v) => { st.l = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 420, H = 290, m = { l: 40, r: 10, t: 24, b: 34 };
  const svgR = NB.svg(row, W, H), svgL = NB.svg(row, W, H);
  const colors = [NB.c.blue, NB.c.orange, NB.c.green, NB.c.pink, NB.c.purple, NB.c.yellow];
  NB.legend(el, beta.map((b, j) => [`β${j + 1} (true ${b})`, colors[j]]));
  const read = NB.readout(el);
  const Xl = NB.scale(L0, L1, m.l, W - m.r), Yb = NB.scale(-2, 2.5, H - m.b, m.t);
  function panel(svg, path, title, cur) {
    NB.clear(svg);
    NB.axes(svg, Xl, Yb, { xlabel: "λ (log scale)", ylabel: "coefficient", xticks: 4, yticks: 5, xfmt: (t) => NB.fmt(10 ** t, 2) });
    NB.el("text", { x: m.l + 4, y: 14, class: "nb-label", style: { fontWeight: 600 } }, svg, title);
    for (let j = 0; j < p; j++) {
      NB.el("path", { d: NB.line(ls.map((l, k) => [l, path[k][j]]), Xl, Yb), fill: "none", stroke: colors[j], "stroke-width": 2 }, svg);
      NB.el("circle", { cx: Xl(st.l), cy: Yb(cur[j]), r: 3.5, fill: colors[j] }, svg);
    }
    NB.el("line", { x1: Xl(st.l), x2: Xl(st.l), y1: Yb(-2), y2: Yb(2.5), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, svg);
  }
  function draw() {
    const lam = 10 ** st.l, r = NB.lstsq(X, y, lam), la = lasso(lam);
    panel(svgR, ridgePath, "ridge (L2)", r);
    panel(svgL, lassoPath, "lasso (L1)", la);
    const nz = la.filter((b) => Math.abs(b) > 1e-8).length;
    read.innerHTML = `λ = ${NB.fmt(lam, 1)}: lasso keeps <b>${nz}</b> of ${p} features; ridge keeps all ${p}. ` +
      `‖β‖₂ ridge <b>${NB.fmt(Math.hypot(...r), 2)}</b>, ‖β‖₁ lasso <b>${NB.fmt(la.reduce((s, b) => s + Math.abs(b), 0), 2)}</b>`;
  }
  draw();
});

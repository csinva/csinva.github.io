/* gradient-descent: contour plot of a 2-d loss; click to choose a start point, sliders set the step size
 * (and momentum / conditioning). Shows the path and the loss per step.
 * params: {surface: "bowl" | "rosenbrock", lr: 0.1, momentum: 0, cond: 5, steps: 40} */
NB.register("gradient-descent", function (el, P) {
  const surfaces = {
    bowl: { // f = (x^2 + k y^2) / 2: k is the condition number of the Hessian
      f: (x, y, k) => 0.5 * (x * x + k * y * y), g: (x, y, k) => [x, k * y], box: [-3, 3, -2, 2], start: [-2.6, 1.6], lrMax: 1.2,
    },
    rosenbrock: { // a curved valley with its minimum at (1, 1)
      f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
      box: [-2, 2, -1, 3], start: [-1.5, 2.5], lrMax: 0.06,
    },
  };
  const S = surfaces[P.surface || "bowl"];
  const st = { lr: P.lr || 0.1, mom: P.momentum || 0, k: P.cond || 5, start: S.start.slice(), steps: P.steps || 40 };
  const W = 620, H = 400, m = { l: 44, r: 12, t: 10, b: 34 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "step size", min: 0.005, max: S.lrMax, step: 0.005, value: st.lr }, (v) => { st.lr = v; draw(); });
  NB.slider(ctl, { label: "momentum", min: 0, max: 0.95, step: 0.05, value: st.mom }, (v) => { st.mom = v; draw(); });
  if (P.surface !== "rosenbrock") NB.slider(ctl, { label: "condition number", min: 1, max: 20, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; drawContours(); draw(); });
  const row = NB.el("div", { class: "nb-row", style: { display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)", gap: "14px" } }, el);
  const svg = NB.svg(row, W, H);
  const svg2 = NB.svg(row, 300, H);
  const read = NB.readout(el);
  const [x0, x1, y0, y1] = S.box;
  const X = NB.scale(x0, x1, m.l, W - m.r), Y = NB.scale(y0, y1, H - m.b, m.t);
  const gC = NB.el("g", {}, svg), gA = NB.el("g", {}, svg), gP = NB.el("g", {}, svg);
  NB.axes(gA, X, Y, { grid: false, xlabel: "θ₁", ylabel: "θ₂" });
  NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", style: { cursor: "crosshair" } }, svg)
    .addEventListener("click", (e) => {
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      st.start = [X.inv(p.x), Y.inv(p.y)]; draw();
    });

  // contours by marching squares on a grid
  function drawContours() {
    NB.clear(gC);
    const nx = 90, ny = 60, f = (a, b) => S.f(a, b, st.k), vals = [];
    for (let j = 0; j <= ny; j++) { vals.push([]); for (let i = 0; i <= nx; i++) vals[j].push(f(x0 + (i / nx) * (x1 - x0), y0 + (j / ny) * (y1 - y0))); }
    const flat = vals.flat().sort((a, b) => a - b);
    const levels = NB.range(0, 1, 12).slice(1).map((q) => flat[Math.floor(q ** 2.2 * (flat.length - 1))]);
    levels.forEach((lv, li) => {
      let d = "";
      const P2 = (i, j) => [X(x0 + (i / nx) * (x1 - x0)), Y(y0 + (j / ny) * (y1 - y0))];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const c = [vals[j][i], vals[j][i + 1], vals[j + 1][i + 1], vals[j + 1][i]];
        const corners = [P2(i, j), P2(i + 1, j), P2(i + 1, j + 1), P2(i, j + 1)];
        const pts = [];
        for (let e = 0; e < 4; e++) {
          const a = c[e], b = c[(e + 1) % 4];
          if ((a - lv) * (b - lv) < 0) { const t = (lv - a) / (b - a), A = corners[e], B = corners[(e + 1) % 4]; pts.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); }
        }
        if (pts.length >= 2) d += `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}L${pts[1][0].toFixed(1)},${pts[1][1].toFixed(1)}`;
        if (pts.length === 4) d += `M${pts[2][0].toFixed(1)},${pts[2][1].toFixed(1)}L${pts[3][0].toFixed(1)},${pts[3][1].toFixed(1)}`;
      }
      NB.el("path", { d, fill: "none", stroke: NB.c.blue, "stroke-opacity": 0.18 + 0.5 * (1 - li / levels.length), "stroke-width": 1 }, gC);
    });
  }

  function draw() {
    NB.clear(gP); NB.clear(svg2);
    let [x, y] = st.start, v = [0, 0];
    const path = [[x, y]], loss = [S.f(x, y, st.k)];
    for (let i = 0; i < st.steps; i++) {
      const g = S.g(x, y, st.k);
      v = [st.mom * v[0] - st.lr * g[0], st.mom * v[1] - st.lr * g[1]];
      x += v[0]; y += v[1];
      if (!isFinite(x) || Math.abs(x) > 1e6) break;
      path.push([x, y]); loss.push(S.f(x, y, st.k));
    }
    const clip = (p) => [Math.max(x0, Math.min(x1, p[0])), Math.max(y0, Math.min(y1, p[1]))];
    NB.el("path", { d: NB.line(path.map(clip), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, gP);
    path.forEach((p, i) => { const q = clip(p); NB.el("circle", { cx: X(q[0]), cy: Y(q[1]), r: i ? 2.6 : 5, fill: i ? NB.c.orange : NB.c.fg }, gP); });
    // loss per step
    const mm = { l: 44, r: 8, t: 10, b: 34 };
    const lmax = Math.max(...loss.slice(0, 3), 1e-6), Ls = loss.map((l) => Math.min(l, lmax * 1.5));
    const XX = NB.scale(0, st.steps, mm.l, 300 - mm.r), YY = NB.scale(0, lmax * 1.5, H - mm.b, mm.t);
    NB.axes(svg2, XX, YY, { xlabel: "step", ylabel: "loss", xticks: 4, yticks: 4 });
    NB.el("path", { d: NB.line(Ls.map((l, i) => [i, l]), XX, YY), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg2);
    const last = loss[loss.length - 1], diverged = loss.length < st.steps + 1 || last > loss[0] * 1.01;
    read.innerHTML = diverged ? `<b>diverges</b>: the step size is too large for the steepest direction`
      : `after ${st.steps} steps: loss <b>${NB.fmt(last, 4)}</b> (start ${NB.fmt(loss[0], 2)}). Click the plot to start somewhere else.`;
  }
  drawContours(); draw();
});

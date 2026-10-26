/* svd-ellipse: X = U Σ Vᵀ for a 2x2 matrix, shown as what each factor does to the unit circle.
 * Vᵀ rotates (green, as in the chapter), Σ stretches along the axes (orange), U rotates back (blue).
 * The thick arc marks the first quarter of the circle so rotations and reflections are visible.
 * params: {a: 1.2, b: 0.8, c: 0.3, d: 0.9} for the matrix [[a, b], [c, d]] */
NB.register("svd-ellipse", function (el, P) {
  const A = [[P.a ?? 1.2, P.b ?? 0.8], [P.c ?? 0.3, P.d ?? 0.9]];
  const ctl = NB.controls(el);
  [["a (row 1, col 1)", 0, 0], ["b (row 1, col 2)", 0, 1], ["c (row 2, col 1)", 1, 0], ["d (row 2, col 2)", 1, 1]].forEach(([lab, i, j]) =>
    NB.slider(ctl, { label: lab, min: -2, max: 2, step: 0.1, value: A[i][j], format: (v) => v.toFixed(1) }, (v) => { A[i][j] = v; draw(); }));
  const grid = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "8px" } }, el);
  const titles = [["x on the unit circle", "fg"], ["Vᵀx (rotate)", "green"], ["ΣVᵀx (stretch)", "orange"], ["UΣVᵀx = Xx", "blue"]];
  const S = 220;
  const panels = titles.map(() => NB.svg(grid, S, S + 22));
  const read = NB.readout(el);

  function svd2(M) {
    const p = M[0][0] ** 2 + M[1][0] ** 2, r = M[0][1] ** 2 + M[1][1] ** 2, q = M[0][0] * M[0][1] + M[1][0] * M[1][1];
    const phi = 0.5 * Math.atan2(2 * q, p - r), mid = (p + r) / 2, rad = Math.hypot((p - r) / 2, q);
    const s1 = Math.sqrt(Math.max(mid + rad, 0)), s2 = Math.sqrt(Math.max(mid - rad, 0));
    const v1 = [Math.cos(phi), Math.sin(phi)], v2 = [-Math.sin(phi), Math.cos(phi)];
    const mv = (v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
    const u1 = s1 > 1e-9 ? mv(v1).map((x) => x / s1) : [1, 0];
    const u2 = s2 > 1e-6 ? mv(v2).map((x) => x / s2) : [-u1[1], u1[0]];
    return { s1, s2, v1, v2, u1, u2 };
  }
  function draw() {
    const { s1, s2, v1, v2, u1, u2 } = svd2(A);
    const Vt = (x) => [v1[0] * x[0] + v1[1] * x[1], v2[0] * x[0] + v2[1] * x[1]];
    const Sg = (x) => [s1 * x[0], s2 * x[1]];
    const U = (x) => [u1[0] * x[0] + u2[0] * x[1], u1[1] * x[0] + u2[1] * x[1]];
    const maps = [(x) => x, Vt, (x) => Sg(Vt(x)), (x) => U(Sg(Vt(x)))];
    const vecs = [[v1, v2], [[1, 0], [0, 1]], [[s1, 0], [0, s2]], [[s1 * u1[0], s1 * u1[1]], [s2 * u2[0], s2 * u2[1]]]];
    const L = Math.max(1.4, s1 * 1.12);
    const circle = NB.range(0, 2 * Math.PI, 181).map((t) => [Math.cos(t), Math.sin(t)]);
    const quarter = NB.range(0, Math.PI / 2, 46).map((t) => [Math.cos(t), Math.sin(t)]);
    panels.forEach((svg, k) => {
      NB.clear(svg);
      const X = NB.scale(-L, L, 6, S - 6), Y = NB.scale(-L, L, S + 16, 22);
      const col = NB.c[titles[k][1]];
      NB.el("text", { x: S / 2, y: 13, "text-anchor": "middle", style: { fontSize: "15px", fill: col, fontWeight: 600 } }, svg, titles[k][0]);
      NB.el("rect", { x: 6, y: 22, width: S - 12, height: S - 6, fill: "none", stroke: NB.c.grid }, svg);
      NB.el("line", { x1: X(-L), x2: X(L), y1: Y(0), y2: Y(0), class: "nb-axis" }, svg);
      NB.el("line", { y1: Y(-L), y2: Y(L), x1: X(0), x2: X(0), class: "nb-axis" }, svg);
      if (k > 0) NB.el("path", { d: NB.line(circle, X, Y), fill: "none", stroke: NB.c.muted, "stroke-dasharray": "3 3", "stroke-opacity": 0.6 }, svg);
      const path = (pts) => "M" + pts.map((p) => { const q = maps[k](p); return X(q[0]).toFixed(1) + "," + Y(q[1]).toFixed(1); }).join("L");
      NB.el("path", { d: path(circle) + "Z", fill: col, "fill-opacity": 0.1, stroke: col, "stroke-width": 1.6 }, svg);
      NB.el("path", { d: path(quarter), fill: "none", stroke: NB.c.fg, "stroke-width": 3.2, "stroke-linecap": "round" }, svg);
      vecs[k].forEach((v, i) => {
        NB.el("line", { x1: X(0), y1: Y(0), x2: X(v[0]), y2: Y(v[1]), stroke: col, "stroke-width": 2.2 }, svg);
        NB.el("circle", { cx: X(v[0]), cy: Y(v[1]), r: 3.2, fill: col }, svg);
        const j = "₁₂"[i], lab = k === 0 ? `v${j}` : k === 1 ? `e${j}` : k === 2 ? `σ${j}e${j}` : `σ${j}u${j}`;
        NB.el("text", { x: X(v[0] * 1.08) + 3, y: Y(v[1] * 1.08) - 3, style: { fontSize: "15px", fill: col } }, svg, lab);
      });
    });
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    read.innerHTML = `σ₁ = <b>${NB.fmt(s1, 2)}</b>, σ₂ = <b>${NB.fmt(s2, 2)}</b> · |det X| = σ₁σ₂ = <b>${NB.fmt(Math.abs(det), 2)}</b> (area of the ellipse over area of the circle)` +
      ` · condition number σ₁/σ₂ = <b>${s2 > 1e-6 ? NB.fmt(s1 / s2, 2) : "∞ (singular)"}</b>` + (det < -1e-9 ? " · det < 0: U and V together include a reflection" : "");
  }
  draw();
});

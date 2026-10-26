/* hat-influence: the diagonal of the hat matrix H = X (X^T X)^{-1} X^T for a line fit y = b0 + b1 x.
 * Twelve fixed points plus one point the reader moves. Each point's area is proportional to H_ii. Since y_hat = H y,
 * raising y_i by 1 raises its own fitted value by H_ii, so a point with H_ii near 1 pulls the line through itself.
 * The dashed line is the fit without the moving point. With an intercept, H_ii = 1/n + (x_i - xbar)^2 / Sxx.
 * params: {x: 4, y: 5} */
NB.register("hat-influence", function (el, P) {
  const rng = NB.rng(29);
  for (let k = 0; k < 5; k++) rng();
  const base = Array.from({ length: 12 }, (_, i) => {
    const x = 1 + (6 * i) / 11 + 0.3 * (rng() - 0.5);
    return [x, 1 + 0.6 * x + 0.6 * rng.normal()];
  });
  const st = { x: P.x ?? 4, y: P.y ?? 5 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "x of the pink point", min: 0, max: 14, step: 0.1, value: st.x }, (v) => { st.x = v; draw(); });
  NB.slider(ctl, { label: "y of the pink point", min: -2, max: 14, step: 0.1, value: st.y }, (v) => { st.y = v; draw(); });
  const W = 640, Hh = 320, m = { l: 44, r: 14, t: 12, b: 36 };
  const svg = NB.svg(el, W, Hh);
  NB.legend(el, [["fit with all 13 points", NB.c.blue], ["fit without the pink point", NB.c.muted, true]]);
  const read = NB.readout(el);
  const cid = "hat-clip-" + Math.random().toString(36).slice(2, 8);
  const X = NB.scale(0, 14, m.l, W - m.r), Y = NB.scale(-2, 14, Hh - m.b, m.t);

  function fit(pts) {                                    // least squares line and the hat diagonal
    const n = pts.length, xb = pts.reduce((s, p) => s + p[0], 0) / n, yb = pts.reduce((s, p) => s + p[1], 0) / n;
    const sxx = pts.reduce((s, p) => s + (p[0] - xb) ** 2, 0), sxy = pts.reduce((s, p) => s + (p[0] - xb) * (p[1] - yb), 0);
    const b1 = sxy / sxx, b0 = yb - b1 * xb;
    return { b0, b1, h: pts.map((p) => 1 / n + (p[0] - xb) ** 2 / sxx) };
  }
  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "x", ylabel: "y", xticks: 7, yticks: 8 });
    const pts = base.concat([[st.x, st.y]]), all = fit(pts), wo = fit(base);
    const clip = (b0, b1) => [[0, b0], [14, b0 + 14 * b1]];
    const cp = NB.el("clipPath", { id: cid }, NB.el("defs", {}, svg));
    NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: Hh - m.t - m.b }, cp);
    NB.el("path", { d: NB.line(clip(wo.b0, wo.b1), X, Y), stroke: NB.c.muted, "stroke-width": 1.6, "stroke-dasharray": "6 4", fill: "none", "clip-path": `url(#${cid})` }, svg);
    NB.el("path", { d: NB.line(clip(all.b0, all.b1), X, Y), stroke: NB.c.blue, "stroke-width": 2.4, fill: "none", "clip-path": `url(#${cid})` }, svg);
    const r = (h) => 2.5 + 15 * Math.sqrt(h);
    pts.forEach((p, i) => {
      const me = i === pts.length - 1;
      NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: r(all.h[i]), fill: me ? NB.c.pink : NB.c.fg, "fill-opacity": me ? 0.55 : 0.25, stroke: me ? NB.c.pink : NB.c.muted, "stroke-width": 1.2 }, svg);
    });
    const hi = all.h[pts.length - 1], yhat = all.b0 + all.b1 * st.x;
    NB.el("line", { x1: X(st.x), x2: X(st.x), y1: Y(st.y), y2: Y(yhat), stroke: NB.c.pink, "stroke-width": 1.5, "stroke-dasharray": "2 2" }, svg);
    const right = st.x > 10.5;
    NB.el("text", { x: X(st.x) + (right ? -1 : 1) * (r(hi) + 6), y: Y(st.y) + 4, "text-anchor": right ? "end" : "start", class: "nb-label", style: { fill: "var(--nb-c-pink)", fontWeight: 600, paintOrder: "stroke", stroke: "var(--nb-bg)", strokeWidth: 4 } }, svg, `H = ${NB.fmt(hi, 2)}`);
    read.innerHTML = `Pink point: H<sub>ii</sub> = <b>${NB.fmt(hi, 3)}</b>, so raising its y by 1 raises its fitted value by ${NB.fmt(hi, 3)}. ` +
      `Fitted value <b>${NB.fmt(yhat, 2)}</b> vs y = ${NB.fmt(st.y, 2)}. ` +
      `Slope <b>${NB.fmt(all.b1, 3)}</b> with it, ${NB.fmt(wo.b1, 3)} without. ` +
      `Largest H<sub>ii</sub> among the gray points: ${NB.fmt(Math.max(...all.h.slice(0, -1)), 3)}.`;
  }
  draw();
});

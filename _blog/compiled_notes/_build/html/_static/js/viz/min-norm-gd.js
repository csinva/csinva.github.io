/* min-norm-gd: gradient descent on underdetermined least squares, one data point a = (2, 1), y = 2, weights
 * w in R^2. Every w on the line a.w = y fits exactly. GD on 0.5 (a.w - y)^2 moves only along a (the row space of
 * X), so from w0 it lands on  w0 + a (y - a.w0) / |a|^2 , the point of the line closest to w0; from w0 = 0 that is
 * the min-norm solution w* = a y / |a|^2. Ridge from 0: w_lam = a y / (|a|^2 + lam), shown at lam = 1 / (eta t).
 * Iterates are computed exactly by the update rule; the end points match numpy's pinv(X) y + (I - pinv(X) X) w0.
 * params: {w0x: 0, w0y: 0, steps: 20} */
NB.register("min-norm-gd", function (el, P) {
  const a = [2, 1], y = 2, aa = 5, eta = 0.04;
  const st = { x0: P.w0x || 0, y0: P.w0y || 0, t: P.steps === undefined ? 20 : P.steps };
  const wstar = [a[0] * y / aa, a[1] * y / aa];
  const nrm = (v) => Math.hypot(v[0], v[1]);

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "init w₀ (first coord.)", min: -1.5, max: 1.5, step: 0.05, value: st.x0 }, (v) => { st.x0 = v; draw(); });
  NB.slider(ctl, { label: "init w₀ (second coord.)", min: -1.5, max: 1.5, step: 0.05, value: st.y0 }, (v) => { st.y0 = v; draw(); });
  NB.slider(ctl, { label: "GD steps t", min: 0, max: 60, step: 1, value: st.t, format: (v) => v.toFixed(0) }, (v) => { st.t = v; draw(); });

  const W = 460, H = 400, m = { l: 40, r: 12, t: 10, b: 36 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["exact fits a·w = y", NB.c.blue], ["GD iterates", NB.c.orange], ["min-norm solution w*", NB.c.green],
    ["ridge from 0 at λ = 1/(ηt)", NB.c.purple], ["circle of radius ‖w*‖", NB.c.muted, true]]);
  const read = NB.readout(el);
  const X = NB.scale(-1.7, 2.1, m.l, W - m.r), Y = NB.scale(-1.6, 2.2, H - m.b, m.t);
  const uid = "mn" + Math.random().toString(36).slice(2, 8);

  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xticks: 4, yticks: 4, xlabel: "w₁", ylabel: "w₂" });
    const cp = NB.el("clipPath", { id: uid }, NB.el("defs", {}, svg));
    NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, cp);
    const g = NB.el("g", { "clip-path": `url(#${uid})` }, svg);
    // level sets of the loss are lines parallel to the solution line
    [-3, -2, -1, 1, 2, 3, 4].forEach((c) => {
      const yy = y + c * 0.9;   // a.w = yy
      NB.el("line", { x1: X(-3), y1: Y((yy - a[0] * -3) / a[1]), x2: X(3), y2: Y((yy - a[0] * 3) / a[1]), stroke: NB.c.muted, "stroke-opacity": 0.25 }, g);
    });
    NB.el("line", { x1: X(-3), y1: Y((y - a[0] * -3) / a[1]), x2: X(3), y2: Y((y - a[0] * 3) / a[1]), stroke: NB.c.blue, "stroke-width": 2.5 }, g);
    // the row space of X: the direction of a through the origin
    NB.el("line", { x1: X(-1.6 * 0.6), y1: Y(-1.6 * 0.3), x2: X(2 * 0.9), y2: Y(2 * 0.45), stroke: NB.c.muted, "stroke-dasharray": "2 4" }, g);
    NB.el("circle", { cx: X(0), cy: Y(0), r: X(nrm(wstar)) - X(0), fill: "none", stroke: NB.c.muted, "stroke-dasharray": "5 4" }, g);
    // GD iterates
    let w = [st.x0, st.y0];
    const pts = [w.slice()];
    for (let k = 0; k < st.t; k++) {
      const r = a[0] * w[0] + a[1] * w[1] - y;
      w = [w[0] - eta * r * a[0], w[1] - eta * r * a[1]];
      pts.push(w.slice());
    }
    NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 1.6 }, g);
    pts.forEach((p, i) => NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: i === pts.length - 1 ? 5 : 2.5, fill: NB.c.orange }, g));
    const lim = [st.x0 + a[0] * (y - a[0] * st.x0 - a[1] * st.y0) / aa, st.y0 + a[1] * (y - a[0] * st.x0 - a[1] * st.y0) / aa];
    NB.el("circle", { cx: X(lim[0]), cy: Y(lim[1]), r: 6, fill: "none", stroke: NB.c.orange, "stroke-width": 1.5 }, g);
    NB.el("circle", { cx: X(wstar[0]), cy: Y(wstar[1]), r: 5.5, fill: NB.c.green }, g);
    if (st.t > 0) {
      const lamb = 1 / (eta * st.t), s = y / (aa + lamb);
      NB.el("rect", { x: X(a[0] * s) - 4, y: Y(a[1] * s) - 4, width: 8, height: 8, fill: NB.c.purple }, g);
    }
    NB.el("circle", { cx: X(st.x0), cy: Y(st.y0), r: 4, fill: NB.c.bg, stroke: NB.c.fg }, g);
    const loss = 0.5 * Math.pow(a[0] * w[0] + a[1] * w[1] - y, 2);
    const off = Math.abs(a[1] * st.x0 - a[0] * st.y0) / Math.sqrt(aa);   // part of w0 orthogonal to a
    read.innerHTML = `after t = ${st.t} steps: loss <b>${NB.fmt(loss, 4)}</b>, ‖w‖ = <b>${NB.fmt(nrm(w), 3)}</b>. ` +
      `GD converges to the open orange circle, norm <b>${NB.fmt(nrm(lim), 3)}</b>; ‖w*‖ = <b>${NB.fmt(nrm(wstar), 3)}</b>. ` +
      (off < 1e-9 ? "w₀ lies in the row space of X, so the limit is w*." :
        `w₀ has a component of size ${NB.fmt(off, 3)} outside the row space, which GD never changes.`);
  }
  draw();
});

/* oja-rule: one linear neuron v = w·x learning from 2-d correlated inputs, one sample at a time.
 * Hebb:  Δw = ε x v              (w turns toward the principal eigenvector but its length grows without bound)
 * Oja:   Δw = ε (x v − α v² w)   (same direction, length settles at 1/√α)
 * The weight path is drawn over the input cloud; the dashed line is the principal eigenvector of the input
 * correlation matrix.
 * params: {rule: "oja", rho: 0.8, alpha: 1, steps: 400} */
NB.register("oja-rule", function (el, P) {
  const st = { rule: P.rule || "oja", rho: P.rho === undefined ? 0.8 : P.rho, alpha: P.alpha || 1, steps: P.steps || 400, seed: 5 };
  const EPS = 0.01, LIM = 3.2;
  const ctl = NB.controls(el);
  NB.select(ctl, "rule", [["oja", "Oja (Hebb + decay)"], ["hebb", "plain Hebb"]], st.rule, (v) => { st.rule = v; draw(); });
  NB.slider(ctl, { label: "input correlation", min: -0.95, max: 0.95, step: 0.05, value: st.rho }, (v) => { st.rho = v; draw(); });
  NB.slider(ctl, { label: "decay α", min: 0.25, max: 4, step: 0.25, value: st.alpha }, (v) => { st.alpha = v; draw(); });
  NB.slider(ctl, { label: "samples seen", min: 0, max: 1500, step: 10, value: st.steps, format: (v) => v.toFixed(0) }, (v) => { st.steps = v; draw(); });
  NB.button(ctl, "new samples", () => { st.seed += 1; draw(); });
  const W = 420, H = 420, m = { l: 40, r: 12, t: 12, b: 36 };
  const svg = NB.svg(el, W, H);
  svg.style.maxWidth = "460px";
  NB.legend(el, [["inputs x", NB.c.muted], ["weight path", NB.c.blue], ["final w", NB.c.orange], ["principal eigenvector", NB.c.green, true]]);
  const read = NB.readout(el);
  const X = NB.scale(-LIM, LIM, m.l, W - m.r), Y = NB.scale(-LIM, LIM, H - m.b, m.t);
  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "x₁", ylabel: "x₂", xticks: 6, yticks: 6 });
    const u = NB.rng(17 + st.seed), r = st.rho, c = Math.sqrt(1 - r * r);
    const sample = () => { const a = u.normal(), b = u.normal(); return [a, r * a + c * b]; };
    for (let i = 0; i < 250; i++) { const p = sample(); if (Math.abs(p[0]) < LIM && Math.abs(p[1]) < LIM) NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 1.8, fill: NB.c.muted, "fill-opacity": 0.35 }, svg); }
    // principal eigenvector of [[1, r], [r, 1]]: (1, 1)/√2 if r > 0, (1, −1)/√2 if r < 0
    const e = r >= 0 ? [Math.SQRT1_2, Math.SQRT1_2] : [Math.SQRT1_2, -Math.SQRT1_2];
    if (Math.abs(r) >= 0.05) NB.el("line", { x1: X(-LIM * e[0]), y1: Y(-LIM * e[1]), x2: X(LIM * e[0]), y2: Y(LIM * e[1]), stroke: NB.c.green, "stroke-dasharray": "6 4", "stroke-width": 1.6 }, svg);
    if (st.rule === "oja") NB.el("circle", { cx: X(0), cy: Y(0), r: X(1 / Math.sqrt(st.alpha)) - X(0), fill: "none", stroke: NB.c.orange, "stroke-opacity": 0.4, "stroke-dasharray": "2 3" }, svg);
    const wu = NB.rng(99 + st.seed);
    let w = [0.6 * (wu() - 0.5), 0.6 * (wu() - 0.5) + 0.25];
    const path = [w.slice()];
    for (let k = 0; k < st.steps; k++) {
      const x = sample(), v = w[0] * x[0] + w[1] * x[1];
      const dec = st.rule === "oja" ? st.alpha * v * v : 0;
      w = [w[0] + EPS * (x[0] * v - dec * w[0]), w[1] + EPS * (x[1] * v - dec * w[1])];
      if (!isFinite(w[0]) || Math.hypot(w[0], w[1]) > 1e9) break;
      path.push(w.slice());
    }
    const clip = (p) => { const n = Math.hypot(p[0], p[1]), s = n > LIM ? LIM / n : 1; return [p[0] * s, p[1] * s]; };
    NB.el("path", { d: NB.line(path.map(clip), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 1.6 }, svg);
    const f = clip(w);
    NB.el("line", { x1: X(0), y1: Y(0), x2: X(f[0]), y2: Y(f[1]), stroke: NB.c.orange, "stroke-width": 3 }, svg);
    NB.el("circle", { cx: X(f[0]), cy: Y(f[1]), r: 5, fill: NB.c.orange }, svg);
    const n = Math.hypot(w[0], w[1]), ang = n > 0 ? (Math.acos(Math.min(1, Math.abs(w[0] * e[0] + w[1] * e[1]) / n)) * 180) / Math.PI : 90;
    read.innerHTML = `|w| = <b>${NB.fmt(n, 2)}</b>` + (st.rule === "oja" ? ` (Oja predicts 1/√α = ${NB.fmt(1 / Math.sqrt(st.alpha), 2)})` : " (no limit: keeps growing)") +
      (Math.abs(r) < 0.05 ? "; with no correlation every direction has the same variance, so there is no principal direction"
        : `; angle to principal eigenvector <b>${NB.fmt(ang, 1)}°</b>`) + (n > LIM ? "; arrow clipped to the plot" : "");
  }
  draw();
});

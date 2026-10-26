/* ridgeless-risk: limiting risk of min-norm (ridgeless) least squares vs gamma = p/n (hastie et al. 2019).
 *   R(g) = s2 * g / (1 - g)                    for g < 1
 *   R(g) = r2 * (1 - 1/g) + s2 / (g - 1)       for g > 1
 * where r2 = ||beta||^2 (signal) and s2 = sigma^2 (noise). Checked against a numpy simulation of
 * pinv(X) y with isotropic Gaussian X (n = 200): matches within simulation error.
 * params: {gamma: 2, r2: 1, s2: 0.25} */
NB.register("ridgeless-risk", function (el, P) {
  const st = { g: P.gamma || 2, r2: P.r2 === undefined ? 1 : P.r2, s2: P.s2 === undefined ? 0.25 : P.s2 };
  const gmax = 5, ymax = 3;
  const risk = (g) => (g < 1 ? (st.s2 * g) / (1 - g) : st.r2 * (1 - 1 / g) + st.s2 / (g - 1));
  const uid = "rr" + Math.random().toString(36).slice(2, 8);

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "γ = p / n", min: 0.1, max: gmax, step: 0.05, value: st.g }, (v) => { st.g = v; draw(); });
  NB.slider(ctl, { label: "signal ‖β‖²", min: 0, max: 2, step: 0.05, value: st.r2 }, (v) => { st.r2 = v; draw(); });
  NB.slider(ctl, { label: "noise σ²", min: 0.01, max: 1, step: 0.01, value: st.s2 }, (v) => { st.s2 = v; draw(); });

  const W = 640, H = 320, m = { l: 48, r: 14, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["limiting risk R(γ)", NB.c.blue], ["‖β‖², the limit as γ → ∞", NB.c.orange, true], ["γ = 1, where p = n", NB.c.muted, true]]);
  const read = NB.readout(el);
  const X = NB.scale(0, gmax, m.l, W - m.r), Y = NB.scale(0, ymax, H - m.b, m.t);
  const clip = (v) => Math.min(v, ymax + 0.2);

  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "γ = p / n  (features per sample)", ylabel: "test risk", xticks: 5, yticks: 6 });
    const cp = NB.el("clipPath", { id: uid }, NB.el("defs", {}, svg));
    NB.el("rect", { x: m.l, y: m.t - 2, width: W - m.l - m.r, height: H - m.t - m.b + 2 }, cp);
    const g = NB.el("g", { "clip-path": `url(#${uid})` }, svg);
    NB.el("line", { x1: X(1), x2: X(1), y1: Y(0), y2: Y(ymax), stroke: NB.c.muted, "stroke-dasharray": "4 4" }, g);
    NB.el("line", { x1: X(0), x2: X(gmax), y1: Y(st.r2), y2: Y(st.r2), stroke: NB.c.orange, "stroke-dasharray": "6 4", "stroke-width": 1.6 }, g);
    // the two branches are drawn separately so the line doesn't jump across the pole at gamma = 1
    const left = NB.range(0.001, 0.999, 300).map((x) => [x, clip(risk(x))]);
    const right = NB.range(1.001, gmax, 500).map((x) => [x, clip(risk(x))]);
    [left, right].forEach((pts) => NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.4 }, g));
    const r = risk(st.g);
    if (r <= ymax) NB.el("circle", { cx: X(st.g), cy: Y(r), r: 5, fill: NB.c.pink }, g);
    const best = right.reduce((a, b) => (b[1] < a[1] ? b : a));
    const side = st.g < 1 ? "underparameterized (p < n): the risk is all variance and grows toward the pole at γ = 1"
      : "overparameterized (p > n): the interpolating fit's risk comes down again, toward ‖β‖²";
    read.innerHTML = `γ = ${NB.fmt(st.g, 2)}: R = <b>${NB.fmt(r, 3)}</b>, ${side}. ` +
      `Lowest risk with γ > 1: <b>${NB.fmt(best[1], 3)}</b> at γ ≈ ${NB.fmt(best[0], 2)}` +
      `${best[0] > gmax - 0.05 ? " (still falling at the right edge)" : ""}.`;
  }
  draw();
});

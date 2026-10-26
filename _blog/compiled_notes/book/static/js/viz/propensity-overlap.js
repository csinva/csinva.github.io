/* propensity-overlap: how the overlap between treated and control units shrinks as treatment depends more on x.
 * x ~ N(0, 1), propensity e(x) = P(T = 1 | x) = sigmoid(gamma x). Densities p(x | T = 1) = phi(x) e(x) / P(T = 1)
 * and p(x | T = 0) = phi(x) (1 - e(x)) / P(T = 0), with P(T = 1) = 1/2 by symmetry.
 *   overlap = integral of min(p(x|T=1), p(x|T=0)) dx      (1 - total variation distance; grid integral)
 *   effective sample size of the inverse-propensity weights 1/e(x) on the treated, per n units:
 *     (E[T/e])^2 / E[T/e^2] = 1 / E[1/e(x)] = 1 / (1 + exp(gamma^2 / 2))   (closed form, same for controls)
 * The closed form and the overlap integral were checked against scipy.integrate.quad and a Monte Carlo draw.
 * The rug is one sample of 200 units (fixed uniforms, so moving gamma reassigns treatment smoothly).
 * params: {gamma: 1.5} */
NB.register("propensity-overlap", function (el, P) {
  const st = { g: P.gamma === undefined ? 1.5 : P.gamma };
  const sig = (v) => 1 / (1 + Math.exp(-v)), phi = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  const u = NB.rng(31);
  for (let i = 0; i < 5; i++) u();
  const units = Array.from({ length: 200 }, () => [u.normal(), u()]);
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "selection strength γ", min: 0, max: 4, step: 0.05, value: st.g }, (v) => { st.g = v; draw(); });

  const W = 640, H = 310, m = { l: 48, r: 48, t: 12, b: 74 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["treated p(x | T = 1)", NB.c.blue], ["control p(x | T = 0)", NB.c.orange], ["propensity e(x) (right axis)", NB.c.green],
    ["e(x) < 0.05 or > 0.95", NB.c.muted]]);
  const read = NB.readout(el);
  const X = NB.scale(-3.5, 3.5, m.l, W - m.r), Y = NB.scale(0, 0.9, H - m.b, m.t), Ye = NB.scale(0, 1, H - m.b, m.t);
  const xs = NB.range(-3.5, 3.5, 351);

  function draw() {
    NB.clear(svg);
    const g = st.g, e = (x) => sig(g * x);
    // shade where e(x) is extreme
    if (g > 0) {
      const c = Math.log(0.95 / 0.05) / g;
      [[-3.5, -c], [c, 3.5]].forEach(([a, b]) => { if (b > a && a < 3.5 && b > -3.5)
        NB.el("rect", { x: X(Math.max(a, -3.5)), y: m.t, width: X(Math.min(b, 3.5)) - X(Math.max(a, -3.5)), height: H - m.b - m.t, fill: NB.c.muted, "fill-opacity": 0.13 }, svg); });
    }
    NB.axes(svg, X, Y, { xticks: 7, yticks: 3, ylabel: "density" });
    NB.ticks(0, 1, 2).forEach((t) => NB.el("text", { x: W - m.r + 6, y: Ye(t) + 4, class: "nb-tick", fill: NB.c.green }, svg, NB.fmt(t)));
    const p1 = xs.map((x) => [x, 2 * phi(x) * e(x)]), p0 = xs.map((x) => [x, 2 * phi(x) * (1 - e(x))]);
    const area = (pts, col) => NB.el("path", { d: NB.line(pts, X, Y) + `L${X(3.5)},${Y(0)}L${X(-3.5)},${Y(0)}Z`, fill: col, "fill-opacity": 0.15, stroke: col, "stroke-width": 2.2 }, svg);
    area(p1, NB.c.blue); area(p0, NB.c.orange);
    NB.el("path", { d: NB.line(xs.map((x) => [x, e(x)]), X, Ye), fill: "none", stroke: NB.c.green, "stroke-width": 2, "stroke-dasharray": "6 3" }, svg);
    // rug of one sample: treated above, control below
    let nT = 0;
    units.forEach(([x, v]) => {
      if (x < -3.5 || x > 3.5) return;
      const t = v < e(x); nT += t;
      const y0 = H - m.b + (t ? 24 : 38);
      NB.el("line", { x1: X(x), x2: X(x), y1: y0, y2: y0 + 8, stroke: t ? NB.c.blue : NB.c.orange, "stroke-width": 1.2 }, svg);
    });
    NB.el("text", { x: m.l - 6, y: H - m.b + 32, class: "nb-tick", "text-anchor": "end" }, svg, "T=1");
    NB.el("text", { x: m.l - 6, y: H - m.b + 46, class: "nb-tick", "text-anchor": "end" }, svg, "T=0");
    NB.el("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "nb-label", "text-anchor": "middle" }, svg, "covariate x");
    // numbers
    const dx = xs[1] - xs[0];
    const ov = xs.reduce((s, x) => s + Math.min(2 * phi(x) * e(x), 2 * phi(x) * (1 - e(x))) * dx, 0);
    const ess = 1 / (1 + Math.exp(g * g / 2));
    const c = g > 0 ? Math.log(19) / g : Infinity;
    const ext = g > 0 ? 2 * (1 - 0.5 * (1 + erf(c / Math.SQRT2))) : 0;
    read.innerHTML = `γ = ${NB.fmt(g, 2)}: overlap ∫min(p₁, p₀) = <b>${NB.fmt(ov, 3)}</b>; ` +
      `<b>${NB.fmt(100 * ext, 1)}%</b> of units have e(x) below 0.05 or above 0.95; inverse-propensity weights leave an effective sample size of ` +
      `<b>${NB.fmt(100 * ess, 1)}%</b> of n in each arm (an even split would give 50%).`;
  }
  // Abramowitz-Stegun 7.1.26 (abs. error < 1.5e-7)
  function erf(x) {
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  draw();
});

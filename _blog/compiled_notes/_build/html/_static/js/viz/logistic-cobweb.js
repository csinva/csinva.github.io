/* logistic-cobweb: functional iteration x_{n+1} = f(x_n) for the logistic map f(x) = μ x (1 - x).
 * Left: cobweb plot (go up to f, across to the diagonal, repeat). Right: x_n against n.
 * The readout gives the fixed point, the derivative there, and the period of the long-run orbit.
 * params: {mu: 2.8, x0: 0.2, steps: 60} */
NB.register("logistic-cobweb", function (el, P) {
  const st = { mu: P.mu ?? 2.8, x0: P.x0 ?? 0.2, steps: P.steps || 60 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "μ", min: 0.5, max: 4, step: 0.005, value: st.mu, format: (v) => v.toFixed(3) }, (v) => { st.mu = v; draw(); });
  NB.slider(ctl, { label: "x₀", min: 0.01, max: 0.99, step: 0.01, value: st.x0 }, (v) => { st.x0 = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.5fr)", gap: "14px" } }, el);
  const H = 300, W1 = 300, W2 = 450, m = { l: 40, r: 10, t: 10, b: 34 };
  const svg1 = NB.svg(row, W1, H), svg2 = NB.svg(row, W2, H);
  NB.legend(el, [["f(x) = μx(1 − x)", NB.c.blue], ["diagonal y = x", NB.c.muted, true], ["iterates", NB.c.orange]]);
  const read = NB.readout(el);

  function period(mu) {
    let x = 0.2345;
    for (let i = 0; i < 20000; i++) x = mu * x * (1 - x);
    const orbit = [x];
    for (let i = 0; i < 128; i++) { x = mu * x * (1 - x); orbit.push(x); }
    for (let p = 1; p <= 64; p++) if (orbit.slice(0, 64).every((v, i) => Math.abs(v - orbit[i + p]) < 1e-5)) return p;
    return 0;
  }
  function draw() {
    NB.clear(svg1); NB.clear(svg2);
    const f = (x) => st.mu * x * (1 - x);
    const X = NB.scale(0, 1, m.l, W1 - m.r), Y = NB.scale(0, 1, H - m.b, m.t);
    NB.axes(svg1, X, Y, { xlabel: "xₙ", ylabel: "xₙ₊₁", xticks: 4, yticks: 4 });
    NB.el("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: NB.c.muted, "stroke-dasharray": "4 4" }, svg1);
    NB.el("path", { d: NB.line(NB.range(0, 1, 200).map((x) => [x, f(x)]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.2 }, svg1);
    const xs = [st.x0];
    for (let i = 0; i < st.steps; i++) xs.push(f(xs[i]));
    let d = `M${X(xs[0]).toFixed(1)},${Y(0).toFixed(1)}`;
    for (let i = 0; i < st.steps; i++) d += `L${X(xs[i]).toFixed(1)},${Y(xs[i + 1]).toFixed(1)}L${X(xs[i + 1]).toFixed(1)},${Y(xs[i + 1]).toFixed(1)}`;
    NB.el("path", { d, fill: "none", stroke: NB.c.orange, "stroke-width": 1.2, "stroke-opacity": 0.85 }, svg1);
    const xstar = 1 - 1 / st.mu;
    if (xstar > 0) NB.el("circle", { cx: X(xstar), cy: Y(xstar), r: 4.5, fill: Math.abs(2 - st.mu) < 1 ? NB.c.fg : NB.c.bg, stroke: NB.c.fg, "stroke-width": 2 }, svg1);
    // time series
    const X2 = NB.scale(0, st.steps, m.l, W2 - m.r);
    NB.axes(svg2, X2, Y, { xlabel: "n", ylabel: "xₙ", xticks: 6, yticks: 4 });
    NB.el("path", { d: NB.line(xs.map((x, i) => [i, x]), X2, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 1.4 }, svg2);
    xs.forEach((x, i) => NB.el("circle", { cx: X2(i), cy: Y(x), r: 2.4, fill: NB.c.orange }, svg2));
    const fp = st.mu > 1 ? xstar : 0, dfp = st.mu > 1 ? 2 - st.mu : st.mu, p = period(st.mu);
    const pr = p === 0 ? "no period up to 64: <b>chaotic</b> (or a very long period)" : p === 1 ? "settles on the <b>fixed point</b>" : `settles on a <b>period-${p} orbit</b>`;
    read.innerHTML = `fixed point x* = <b>${NB.fmt(fp, 3)}</b>, f′(x*) = <b>${NB.fmt(dfp, 3)}</b> ` +
      `(${Math.abs(dfp) < 1 ? "|f′| < 1: stable" : "|f′| > 1: unstable"}) · long run: ${pr}`;
  }
  draw();
});

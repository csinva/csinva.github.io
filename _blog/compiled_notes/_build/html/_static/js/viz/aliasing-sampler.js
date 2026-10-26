/* aliasing-sampler: sample cos(2π f t) once per unit of t. Above the Nyquist limit (f = 1/2 cycle per sample)
 * the samples are exactly those of a slower cosine at the folded frequency |f - round(f)|.
 * params: {f: 0.833} */
NB.register("aliasing-sampler", function (el, P) {
  const st = { f: P.f ?? 5 / 6 }, T = 12;
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "signal frequency f (cycles per sample)", min: 0, max: 2, step: 0.01, value: st.f }, (v) => { st.f = v; draw(); });
  const W = 640, H = 260, m = { l: 36, r: 10, t: 12, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["true signal", NB.c.muted], ["samples", NB.c.fg], ["what the samples look like (alias)", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(0, T, m.l, W - m.r), Y = NB.scale(-1.3, 1.3, H - m.b, m.t);
  function draw() {
    NB.clear(svg);
    const f = st.f, fa = Math.abs(f - Math.round(f)), ts = NB.range(0, T, 900);
    NB.axes(svg, X, Y, { grid: false, xlabel: "time (in sample periods)", xticks: 12, yticks: 2 });
    for (let n = 0; n <= T; n++) NB.el("line", { x1: X(n), x2: X(n), y1: Y(-1.3), y2: Y(1.3), stroke: NB.c.grid }, svg);
    NB.el("path", { d: NB.line(ts.map((t) => [t, Math.cos(2 * Math.PI * f * t)]), X, Y), fill: "none", stroke: NB.c.muted, "stroke-width": 1.5 }, svg);
    NB.el("path", { d: NB.line(ts.map((t) => [t, Math.cos(2 * Math.PI * fa * t)]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2.2, "stroke-dasharray": fa === f ? "6 4" : null }, svg);
    for (let n = 0; n <= T; n++) NB.el("circle", { cx: X(n), cy: Y(Math.cos(2 * Math.PI * f * n)), r: 4.5, fill: NB.c.fg }, svg);
    read.innerHTML = f <= 0.5
      ? `f = ${NB.fmt(f, 2)} is below the Nyquist limit of 0.5 cycle per sample: the samples pin down the signal.`
      : `f = ${NB.fmt(f, 2)} is above 0.5 cycle per sample: the samples are identical to those of a cosine at <b>${NB.fmt(fa, 2)}</b> cycle per sample, an <b>alias</b>.`;
  }
  draw();
});

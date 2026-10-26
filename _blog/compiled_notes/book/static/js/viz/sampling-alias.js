/* sampling-alias: sampling a sinusoid of frequency f at rate f_s. Below 2 samples per cycle (f_s < 2f), the
 * same samples are also fit by a slower sinusoid (the alias), so the samples no longer determine the signal.
 * params: {f: 3, fs: 10} */
NB.register("sampling-alias", function (el, P) {
  const st = { f: P.f ?? 3, fs: P.fs ?? 10 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "signal frequency f (Hz)", min: 0.5, max: 12, step: 0.1, value: st.f, format: (v) => v.toFixed(1) }, (v) => { st.f = v; draw(); });
  NB.slider(ctl, { label: "sampling rate fₛ (Hz)", min: 2, max: 30, step: 0.5, value: st.fs, format: (v) => v.toFixed(1) }, (v) => { st.fs = v; draw(); });
  const W = 640, H = 260, m = { l: 44, r: 10, t: 10, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["signal sin(2πft)", NB.c.blue], ["samples, every 1/fₛ seconds", NB.c.orange], ["slowest sinusoid through the samples", NB.c.green, true]]);
  const read = NB.readout(el);

  function draw() {
    NB.clear(svg);
    const T = 1, X = NB.scale(0, T, m.l, W - m.r), Y = NB.scale(-1.3, 1.3, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "time t (seconds)", xticks: 5, yticks: 4 });
    const ts = NB.range(0, T, 1500), k = Math.round(st.f / st.fs), fa = st.f - k * st.fs;
    NB.el("path", { d: NB.line(ts.map((t) => [t, Math.sin(2 * Math.PI * st.f * t)]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 1.6, "stroke-opacity": 0.8 }, svg);
    if (k !== 0 && Math.abs(st.fs - 2 * st.f) > 1e-9) NB.el("path", { d: NB.line(ts.map((t) => [t, Math.sin(2 * Math.PI * fa * t)]), X, Y), fill: "none", stroke: NB.c.green, "stroke-width": 2.4, "stroke-dasharray": "6 4" }, svg);
    for (let n = 0; n / st.fs <= T + 1e-9; n++) {
      const t = n / st.fs;
      NB.el("line", { x1: X(t), x2: X(t), y1: Y(0), y2: Y(Math.sin(2 * Math.PI * st.f * t)), stroke: NB.c.orange, "stroke-opacity": 0.5 }, svg);
      NB.el("circle", { cx: X(t), cy: Y(Math.sin(2 * Math.PI * st.f * t)), r: 4, fill: NB.c.orange }, svg);
    }
    const ok = st.fs > 2 * st.f;
    const edge = Math.abs(st.fs - 2 * st.f) < 1e-9;
    read.innerHTML = `${NB.fmt(st.fs / st.f, 2)} samples per cycle · ` + (edge ? `f<sub>s</sub> = 2f exactly: every sample lands on a zero of the sine` : ok
      ? `f<sub>s</sub> > 2f: <b>no sinusoid slower than f<sub>s</sub>/2 other than the signal fits these samples</b>`
      : `f<sub>s</sub> < 2f: the samples also fit a <b>${NB.fmt(Math.abs(fa), 2)} Hz</b> sinusoid (green), so they can't tell the two apart`);
  }
  draw();
});

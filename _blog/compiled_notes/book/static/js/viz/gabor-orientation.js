/* gabor-orientation: a V1 simple cell as a Gabor receptive field (isotropic Gaussian envelope of width sigma times
 * a cosine carrier of frequency f0 = 1 cycle per unit, bars vertical), shown sinusoidal gratings of the same
 * frequency at orientation theta (angle of the bars from horizontal, 90 = vertical = preferred).
 * Linear response to a grating at its best phase = |G^(k)|, the Fourier transform of the receptive field at the
 * grating's wave vector k:  G^(k) = 1/2 [g(k - k0) + g(k + k0)],  g(k) = exp(-2 pi^2 sigma^2 |k|^2),
 * normalized so the preferred grating gives 1, then half-wave rectified (it is never negative here).
 * Wider envelope (more carrier cycles) -> narrower orientation tuning.
 * params: {theta: 60, sigma: 0.5} */
NB.register("gabor-orientation", function (el, P) {
  const st = { th: P.theta ?? 60, sg: P.sigma ?? 0.5 };
  const f0 = 1, N = 35, EXT = 2;              // image grid: N x N pixels over [-EXT, EXT]
  const rad = (d) => (d * Math.PI) / 180;
  const kvec = (th) => [f0 * Math.sin(rad(th)), -f0 * Math.cos(rad(th))];   // wave vector, perpendicular to the bars
  const g = (kx, ky, s) => Math.exp(-2 * Math.PI * Math.PI * s * s * (kx * kx + ky * ky));
  const resp = (th, s) => {
    const [kx, ky] = kvec(th);
    const R = 0.5 * (g(kx - f0, ky, s) + g(kx + f0, ky, s));
    return R / (0.5 * (1 + g(2 * f0, 0, s)));
  };
  const halfWidth = (s) => {                  // half-width at half height, by bisection on [0, 90]
    let lo = 0, hi = 90;
    if (resp(0, s) >= 0.5) return null;
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (resp(90 + m, s) > 0.5) lo = m; else hi = m; }
    return (lo + hi) / 2;
  };

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "grating orientation θ", min: 0, max: 180, step: 1, value: st.th, format: (v) => v.toFixed(0) + "°" }, (v) => { st.th = v; draw(); });
  NB.slider(ctl, { label: "envelope width σ (carrier periods)", min: 0.2, max: 1.2, step: 0.05, value: st.sg, format: (v) => v.toFixed(2) }, (v) => { st.sg = v; draw(); });

  const W = 660, H = 300, svg = NB.svg(el, W, H);
  NB.legend(el, [["receptive field / grating +", NB.c.orange], ["−", NB.c.blue], ["tuning curve", NB.c.green]]);
  const read = NB.readout(el);

  // two pixel images, built once; draw() only changes fill/opacity
  const PX = 118, cell = PX / N;
  function image(x0, y0, title) {
    NB.el("text", { x: x0 + PX / 2, y: y0 - 6, class: "nb-tick", "text-anchor": "middle" }, svg, title);
    NB.el("rect", { x: x0, y: y0, width: PX, height: PX, fill: "none", stroke: NB.c.grid }, svg);
    const cells = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      cells.push(NB.el("rect", { x: (x0 + j * cell).toFixed(2), y: (y0 + i * cell).toFixed(2), width: (cell + 0.3).toFixed(2), height: (cell + 0.3).toFixed(2), stroke: "none" }, svg));
    return cells;
  }
  const coords = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) coords.push([-EXT + (2 * EXT * (j + 0.5)) / N, EXT - (2 * EXT * (i + 0.5)) / N]);
  const rfCells = image(14, 26, "receptive field"), grCells = image(14, 172, "grating");
  const paint = (cells, vals) => cells.forEach((c, i) => {
    const v = vals[i];
    c.setAttribute("fill", v >= 0 ? NB.c.orange : NB.c.blue);
    c.setAttribute("fill-opacity", Math.min(1, Math.abs(v)).toFixed(3));
  });
  const plot = NB.el("g", {}, svg);
  const m = { l: 200, r: 14, t: 18, b: 44 };
  const X = NB.scale(0, 180, m.l, W - m.r), Y = NB.scale(0, 1.05, H - m.b, m.t);

  function draw() {
    const s = st.sg, [kx, ky] = kvec(st.th);
    paint(rfCells, coords.map(([x, y]) => Math.exp(-(x * x + y * y) / (2 * s * s)) * Math.cos(2 * Math.PI * f0 * x)));
    paint(grCells, coords.map(([x, y]) => Math.cos(2 * Math.PI * (kx * x + ky * y))));
    NB.clear(plot);
    NB.axes(plot, X, Y, { xlabel: "grating orientation θ (90° = preferred, vertical bars)", ylabel: "response", xticks: 6, yticks: 5, xfmt: (t) => t + "°" });
    const ths = NB.range(0, 180, 361);
    const hw = halfWidth(s);
    if (hw !== null) {
      NB.el("rect", { x: X(90 - hw), y: m.t, width: X(90 + hw) - X(90 - hw), height: H - m.t - m.b, fill: NB.c.green, "fill-opacity": 0.08 }, plot);
      NB.el("line", { x1: X(90 - hw), x2: X(90 + hw), y1: Y(0.5), y2: Y(0.5), stroke: NB.c.green, "stroke-dasharray": "4 3" }, plot);
    }
    NB.el("path", { d: NB.line(ths.map((t) => [t, resp(t, s)]), X, Y), fill: "none", stroke: NB.c.green, "stroke-width": 2.5 }, plot);
    const r = resp(st.th, s);
    NB.el("line", { x1: X(st.th), x2: X(st.th), y1: Y(0), y2: Y(r), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, plot);
    NB.el("circle", { cx: X(st.th), cy: Y(r), r: 6, fill: NB.c.green }, plot);
    read.innerHTML = `response to this grating <b>${Math.round(100 * r)}%</b> of the preferred one; ` +
      (hw === null ? "the cell never drops to half its peak (σ too small to be selective)" :
        `tuning half-width at half height <b>±${hw.toFixed(1)}°</b>`) +
      `; about <b>${(2 * s * f0 * 2).toFixed(1)}</b> carrier cycles fit inside ±2σ`;
  }
  draw();
});

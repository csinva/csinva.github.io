/* two-slit-amplitudes: probability on a screen behind two slits. Each slit contributes a complex amplitude;
 * quantum probability is |a1 + a2|^2, while adding probabilities (|a1|^2 + |a2|^2) gives no fringes.
 * Right: the two amplitudes as arrows at the chosen screen position and their sum.
 * params: {sep: 3, wavelength: 0.6, pos: 0} */
NB.register("two-slit-amplitudes", function (el, P) {
  const st = { sep: P.sep || 3, lam: P.wavelength || 0.6, pos: P.pos || 0, watch: false };
  const L = 20, env = 7;                                      // screen distance, width of each slit's bump
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "slit separation", min: 1, max: 6, step: 0.1, value: st.sep, format: (v) => v.toFixed(1) }, (v) => { st.sep = v; draw(); });
  NB.slider(ctl, { label: "wavelength", min: 0.3, max: 1.5, step: 0.05, value: st.lam, format: (v) => v.toFixed(2) }, (v) => { st.lam = v; draw(); });
  NB.slider(ctl, { label: "screen position", min: -12, max: 12, step: 0.1, value: st.pos, format: (v) => v.toFixed(1) }, (v) => { st.pos = v; draw(); });
  const box = NB.el("label", { class: "nb-select" }, ctl);
  const cb = NB.el("input", { type: "checkbox" }, box);
  NB.el("span", { class: "nb-slider-label" }, box, "measure which slit (decoherence)");
  cb.addEventListener("change", () => { st.watch = cb.checked; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 440, H = 260, m = { l: 40, r: 10, t: 10, b: 34 };
  const svg = NB.svg(row, W, H), svg2 = NB.svg(row, 220, H);
  NB.legend(el, [["slit 1 amplitude", NB.c.blue], ["slit 2 amplitude", NB.c.orange], ["|a₁ + a₂|² (amplitudes add)", NB.c.purple], ["|a₁|² + |a₂|² (probabilities add)", NB.c.fg, true]]);
  const read = NB.readout(el);

  function amps(y) {
    const out = [];
    [-st.sep / 2, st.sep / 2].forEach((s) => {
      const r = Math.hypot(L, y - s), mag = Math.exp(-((y - s) ** 2) / (2 * env * env)) / Math.SQRT2, ph = (2 * Math.PI * r) / st.lam;
      out.push([mag * Math.cos(ph), mag * Math.sin(ph)]);
    });
    return out;
  }
  const pQ = (y) => { const [a, b] = amps(y); return (a[0] + b[0]) ** 2 + (a[1] + b[1]) ** 2; };
  const pC = (y) => { const [a, b] = amps(y); return a[0] ** 2 + a[1] ** 2 + b[0] ** 2 + b[1] ** 2; };

  function draw() {
    NB.clear(svg); NB.clear(svg2);
    const X = NB.scale(-12, 12, m.l, W - m.r), Y = NB.scale(0, 2.1, H - m.b, m.t), ys = NB.range(-12, 12, 600);
    NB.axes(svg, X, Y, { xlabel: "position on the screen", ylabel: "probability density", xticks: 6, yticks: 4 });
    NB.el("path", { d: NB.line(ys.map((y) => [y, pC(y)]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.4 }, svg);
    if (!st.watch) NB.el("path", { d: NB.line(ys.map((y) => [y, pQ(y)]), X, Y), fill: "none", stroke: NB.c.purple, "stroke-width": 2 }, svg);
    NB.el("line", { x1: X(st.pos), x2: X(st.pos), y1: Y(0), y2: Y(2.1), stroke: NB.c.pink, "stroke-width": 1.2, "stroke-dasharray": "3 3" }, svg);
    // phasor diagram at the chosen position
    const [a, b] = amps(st.pos), c = 110, cy = 120, s = 70;
    NB.el("circle", { cx: c, cy, r: s / Math.SQRT2, fill: "none", stroke: "var(--nb-grid)" }, svg2);
    NB.el("line", { x1: c - 100, x2: c + 100, y1: cy, y2: cy, class: "nb-grid" }, svg2);
    NB.el("line", { x1: c, x2: c, y1: cy - 100, y2: cy + 100, class: "nb-grid" }, svg2);
    const arrow = (x1, y1, x2, y2, col, w) => {
      NB.el("line", { x1, y1, x2, y2, stroke: col, "stroke-width": w }, svg2);
      const ang = Math.atan2(y2 - y1, x2 - x1), hl = 7;
      if (Math.hypot(x2 - x1, y2 - y1) > 4) NB.el("path", { d: `M${x2},${y2} L${x2 - hl * Math.cos(ang - 0.4)},${y2 - hl * Math.sin(ang - 0.4)} L${x2 - hl * Math.cos(ang + 0.4)},${y2 - hl * Math.sin(ang + 0.4)} Z`, fill: col }, svg2);
    };
    const p1 = [c + s * a[0], cy - s * a[1]], p2 = [p1[0] + s * b[0], p1[1] - s * b[1]];
    if (!st.watch) arrow(c, cy, p2[0], p2[1], NB.c.purple, 4);
    arrow(c, cy, p1[0], p1[1], NB.c.blue, 2);
    arrow(p1[0], p1[1], p2[0], p2[1], NB.c.orange, 2);
    NB.el("text", { x: 110, y: 18, class: "nb-label", "text-anchor": "middle" }, svg2, "amplitudes at the pink line");
    const q = pQ(st.pos), cl = pC(st.pos);
    read.innerHTML = st.watch
      ? `Measuring the slit destroys the interference: probabilities add, <b>${NB.fmt(cl, 3)}</b> at this position.`
      : `At this position |a₁ + a₂|² = <b>${NB.fmt(q, 3)}</b> vs |a₁|² + |a₂|² = <b>${NB.fmt(cl, 3)}</b>. ` +
        (q < 0.15 * cl ? "The two arrows point in opposite directions and cancel." : q > 1.7 * cl ? "The arrows point the same way and reinforce." : "");
  }
  draw();
});

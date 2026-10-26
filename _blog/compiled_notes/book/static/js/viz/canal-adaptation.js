/* canal-adaptation: a horizontal semicircular canal as a high-pass filter of head rotation.
 * The head spins at a constant speed, then stops. The canal signal c follows dc/dt = dω/dt − c/τ:
 * it jumps when the spin starts, fades as the fluid catches up, and swings negative when the head stops
 * (the fluid keeps moving, so the canal reports a turn the other way).
 * params: {spin: 20, tau: 6} */
NB.register("canal-adaptation", function (el, P) {
  const st = { spin: P.spin || 20, tau: P.tau || 6 };
  const T = 60, t0 = 5, dt = 0.02;
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "spin duration (s)", min: 2, max: 40, step: 1, value: st.spin, format: (v) => v.toFixed(0) }, (v) => { st.spin = v; draw(); });
  NB.slider(ctl, { label: "canal time constant τ (s)", min: 1, max: 15, step: 0.5, value: st.tau, format: (v) => v.toFixed(1) }, (v) => { st.tau = v; draw(); });
  const W = 640, H = 300, m = { l: 48, r: 12, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["head rotation speed", NB.c.muted, true], ["canal signal (felt rotation)", NB.c.blue]]);
  const read = NB.readout(el);
  const X = NB.scale(0, T, m.l, W - m.r), Y = NB.scale(-1.1, 1.3, H - m.b, m.t);
  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "time (s)", ylabel: "rightward rotation", xticks: 6, yticks: 4, yfmt: (t) => (t > 1.05 ? "" : NB.fmt(t)) });
    NB.el("line", { x1: X(0), x2: X(T), y1: Y(0), y2: Y(0), stroke: NB.c.muted, "stroke-width": 1 }, svg);
    const t1 = Math.min(T, t0 + st.spin);
    [[0, t0, "still"], [t0, t1, "spinning"], [t1, T, "stopped"]].forEach(([a, b, lab]) => {
      if (b - a > 4) NB.el("text", { x: X((a + b) / 2), y: Y(1.17), class: "nb-tick", "text-anchor": "middle" }, svg, lab);
    });
    const w = (t) => (t >= t0 && t < t1 ? 1 : 0);
    NB.el("path", { d: NB.line([[0, 0], [t0, 0], [t0, 1], [t1, 1], [t1, 0], [T, 0]], X, Y), fill: "none", stroke: NB.c.muted, "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    let c = 0, prev = 0, atStop = 0, after = 0;
    const pts = [[0, 0]];
    const n = Math.round(T / dt);
    for (let i = 1; i <= n; i++) {
      const t = i * dt, cur = w(t), stopping = prev === 1 && cur === 0;
      if (stopping) atStop = c;
      c += cur - prev;            // a step in head speed moves the cupula at once
      c -= (dt / st.tau) * c;     // then the fluid catches up and the cupula relaxes
      if (stopping) after = c;
      prev = cur;
      pts.push([t, c]);
    }
    if (t1 >= T) { atStop = c; after = 0; }
    NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.5 }, svg);
    read.innerHTML = `just before stopping the canal reports <b>${Math.round(100 * atStop)}%</b> of the true spin; ` +
      `right after stopping it reports <b>${Math.round(-100 * after)}%</b> spin the other way`;
  }
  draw();
});

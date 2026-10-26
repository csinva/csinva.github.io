/* char-roots: y'' + p y' + q y = 0 with y(0) = 1, y'(0) = 0. Plugging in e^{rt} gives r² + p r + q = 0.
 * Left: the two roots in the complex plane. Right: the solution, which is a sum of e^{rt} terms
 * (real roots), (c1 + c2 t) e^{rt} (repeated root) or e^{at}(c1 cos bt + c2 sin bt) (roots a ± bi).
 * params: {p: 0.5, q: 4} */
NB.register("char-roots", function (el, P) {
  const st = { p: P.p ?? 0.5, q: P.q ?? 4 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "p (coefficient of y′)", min: -1, max: 5, step: 0.1, value: st.p, format: (v) => v.toFixed(1) }, (v) => { st.p = v; draw(); });
  NB.slider(ctl, { label: "q (coefficient of y)", min: -2, max: 8, step: 0.1, value: st.q, format: (v) => v.toFixed(1) }, (v) => { st.q = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,2fr)", gap: "14px" } }, el);
  const H = 280, svg1 = NB.svg(row, 280, H), svg2 = NB.svg(row, 560, H);
  const read = NB.readout(el);
  const m = { l: 44, r: 10, t: 10, b: 34 };

  function solve(b, c) {
    const disc = b * b - 4 * c;
    if (Math.abs(disc) < 1e-9) { const r = -b / 2; return { kind: "repeated", roots: [[r, 0], [r, 0]], y: (t) => (1 - r * t) * Math.exp(r * t) }; }
    if (disc > 0) {
      const r1 = (-b + Math.sqrt(disc)) / 2, r2 = (-b - Math.sqrt(disc)) / 2, c1 = -r2 / (r1 - r2), c2 = r1 / (r1 - r2);
      return { kind: "real", roots: [[r1, 0], [r2, 0]], y: (t) => c1 * Math.exp(r1 * t) + c2 * Math.exp(r2 * t) };
    }
    const a = -b / 2, w = Math.sqrt(-disc) / 2;
    return { kind: "complex", roots: [[a, w], [a, -w]], a, w, y: (t) => Math.exp(a * t) * (Math.cos(w * t) - (a / w) * Math.sin(w * t)) };
  }
  function draw() {
    NB.clear(svg1); NB.clear(svg2);
    const s = solve(st.p, st.q);
    // roots in the complex plane
    const R = 4, X = NB.scale(-R, R, m.l, 280 - m.r), Y = NB.scale(-R, R, H - m.b, m.t);
    NB.el("rect", { x: m.l, y: m.t, width: X(0) - m.l, height: H - m.b - m.t, fill: NB.c.green, "fill-opacity": 0.07 }, svg1);
    NB.el("rect", { x: X(0), y: m.t, width: 280 - m.r - X(0), height: H - m.b - m.t, fill: NB.c.pink, "fill-opacity": 0.07 }, svg1);
    NB.axes(svg1, X, Y, { xlabel: "real part", ylabel: "imaginary part", xticks: 4, yticks: 4 });
    NB.el("line", { x1: X(0), x2: X(0), y1: Y(-R), y2: Y(R), class: "nb-axis" }, svg1);
    NB.el("line", { x1: X(-R), x2: X(R), y1: Y(0), y2: Y(0), class: "nb-axis" }, svg1);
    NB.el("text", { x: X(-R) + 4, y: m.t + 14, class: "nb-tick" }, svg1, "decays");
    NB.el("text", { x: X(R) - 4, y: m.t + 14, class: "nb-tick", "text-anchor": "end" }, svg1, "grows");
    s.roots.forEach(([re, im], i) => {
      const cx = X(Math.max(-R, Math.min(R, re))), cy = Y(Math.max(-R, Math.min(R, im)));
      NB.el("circle", { cx, cy, r: s.kind === "repeated" ? 7 : 5.5, fill: s.kind === "repeated" ? "none" : NB.c.orange, stroke: NB.c.orange, "stroke-width": 2.5 }, svg1);
    });
    // solution y(t)
    const T = 12, ts = NB.range(0, T, 600), ys = ts.map(s.y);
    const lim = 2.2, X2 = NB.scale(0, T, m.l, 560 - m.r), Y2 = NB.scale(-lim, lim, H - m.b, m.t);
    NB.axes(svg2, X2, Y2, { xlabel: "t", ylabel: "y(t)", xticks: 6, yticks: 4 });
    const clip = ys.map((y, i) => [ts[i], Math.max(-lim * 1.5, Math.min(lim * 1.5, y))]);
    const cp = "cr" + Math.random().toString(36).slice(2, 8);
    NB.el("rect", { x: m.l, y: m.t, width: 560 - m.l - m.r, height: H - m.t - m.b }, NB.el("clipPath", { id: cp }, NB.el("defs", {}, svg2)));
    const g = NB.el("g", { "clip-path": `url(#${cp})` }, svg2);
    if (s.kind === "complex") [1, -1].forEach((sg) => NB.el("path", { d: NB.line(ts.map((t) => [t, sg * Math.exp(s.a * t) * Math.hypot(1, s.a / s.w)]), X2, Y2), fill: "none", stroke: NB.c.muted, "stroke-dasharray": "4 4" }, g));
    NB.el("path", { d: NB.line(clip, X2, Y2), fill: "none", stroke: NB.c.blue, "stroke-width": 2.4 }, g);
    const f = (v) => NB.fmt(v, 2);
    const msg = s.kind === "real" ? `two real roots r = ${f(s.roots[0][0])}, ${f(s.roots[1][0])}: y = c₁e<sup>r₁t</sup> + c₂e<sup>r₂t</sup>, no oscillation`
      : s.kind === "repeated" ? `repeated root r = ${f(s.roots[0][0])}: y = (c₁ + c₂t)e<sup>rt</sup>`
      : `complex roots r = ${f(s.a)} ± ${f(s.w)}i: y = e<sup>${f(s.a)}t</sup>(c₁cos ${f(s.w)}t + c₂sin ${f(s.w)}t); dashed: the envelope, proportional to e<sup>at</sup>`;
    const maxRe = Math.max(s.roots[0][0], s.roots[1][0]);
    read.innerHTML = `p² − 4q = <b>${f(st.p * st.p - 4 * st.q)}</b> · ${msg} · <b>${maxRe < -1e-9 ? "decays to 0" : maxRe > 1e-9 ? "blows up" : "neither decays nor grows"}</b>`;
  }
  draw();
});

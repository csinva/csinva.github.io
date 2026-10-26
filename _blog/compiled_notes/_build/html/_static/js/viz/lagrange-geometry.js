/* lagrange-geometry: minimize f0(x) = (x1 - a)^2 + c (x2 - b)^2 subject to h(x) = x1^2 + x2^2 - 1 = 0.
 * Left: contours of f0 (blue, the note's color for the objective), the constraint circle (green, the color for
 * equality constraints), a point x on it that can be dragged, and the arrows grad f0 and grad h at x.
 * Right: f0 along the circle against the angle. At a stationary point the two gradients are parallel and
 * grad f0 + nu grad h = 0 (KKT condition 5 with only an equality constraint). nu shown is the least-squares
 * value -(grad f0 . grad h) / |grad h|^2, which solves the condition exactly there.
 * Checked against numpy: constrained min at 50.8 degrees (f0 = 1.205), max at 251.3 degrees (f0 = 14.84).
 * params: {a: 1.5, b: 1.2, c: 2.5, theta: 150} */
NB.register("lagrange-geometry", function (el, P) {
  const a = P.a ?? 1.5, b = P.b ?? 1.2, c = P.c ?? 2.5;
  const f0 = (x, y) => (x - a) ** 2 + c * (y - b) ** 2;
  const grad = (x, y) => [2 * (x - a), 2 * c * (y - b)];
  const fth = (t) => f0(Math.cos(t), Math.sin(t));
  const dfth = (t) => { const g = grad(Math.cos(t), Math.sin(t)); return -g[0] * Math.sin(t) + g[1] * Math.cos(t); };
  // stationary points of f0 on the circle: sign changes of df/dtheta, refined by bisection
  const stat = [];
  const N = 720;
  for (let i = 0; i < N; i++) {
    let lo = (2 * Math.PI * i) / N, hi = (2 * Math.PI * (i + 1)) / N;
    if (dfth(lo) * dfth(hi) > 0) continue;
    const up = dfth(lo) < dfth(hi);
    for (let k = 0; k < 60; k++) { const mid = (lo + hi) / 2; if ((dfth(mid) < 0) === up) lo = mid; else hi = mid; }
    stat.push({ t: (lo + hi) / 2, min: up });
  }
  const best = stat.filter((s) => s.min).reduce((p, q) => (fth(q.t) < fth(p.t) ? q : p));
  const deg = (t) => (((t * 180) / Math.PI) % 360 + 360) % 360;
  const angDist = (t, u) => Math.abs(Math.atan2(Math.sin(t - u), Math.cos(t - u)));
  const snap = (t, tol) => { const near = stat.find((p) => angDist(t, p.t) < tol); return near ? near.t : t; };
  const st = { t: ((P.theta ?? 150) * Math.PI) / 180 };

  const ctl = NB.controls(el);
  const sl = NB.slider(ctl, { label: "angle of x on the circle", min: 0, max: 359, step: 1, value: deg(st.t), format: (v) => v.toFixed(0) + "°" }, (v) => { st.t = snap((v * Math.PI) / 180, 0.009); draw(); });
  NB.button(ctl, "jump to the constrained minimum", () => { st.t = best.t; sl.set(Math.round(deg(best.t)) % 360); draw(); });
  const W = 640, H = 330, pw = 340;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["contours of f₀ and ∇f₀", NB.c.blue], ["constraint h(x) = 0 and ∇h", NB.c.green], ["x on the constraint", NB.c.pink]]);
  const read = NB.readout(el);
  const x0 = -2.1, y1 = 2.3, sc = (pw - 10) / 4.6, X = (v) => 10 + (v - x0) * sc, Y = (v) => 12 + (y1 - v) * sc;
  const Xinv = (px) => (px - 10) / sc + x0, Yinv = (py) => y1 - (py - 12) / sc;
  const m2 = { l: pw + 52, r: 14, t: 14, b: 40 };
  const fmax = Math.max(...NB.range(0, 2 * Math.PI, 361).map(fth));
  const Xt = NB.scale(0, 360, m2.l, W - m2.r), Yt = NB.scale(0, Math.ceil(fmax / 4) * 4, H - m2.b, m2.t);

  const arrow = (g, x0, y0, x1, y1, color, w) => {
    const px = [X(x0), Y(y0), X(x1), Y(y1)], ang = Math.atan2(px[3] - px[1], px[2] - px[0]), L = Math.hypot(px[2] - px[0], px[3] - px[1]);
    if (L < 1) return;
    const hl = Math.min(10, L * 0.5);
    NB.el("line", { x1: px[0], y1: px[1], x2: px[2] - Math.cos(ang) * hl * 0.8, y2: px[3] - Math.sin(ang) * hl * 0.8, stroke: color, "stroke-width": w || 2.6 }, g);
    NB.el("polygon", { points: [[px[2], px[3]], [px[2] - hl * Math.cos(ang - 0.4), px[3] - hl * Math.sin(ang - 0.4)], [px[2] - hl * Math.cos(ang + 0.4), px[3] - hl * Math.sin(ang + 0.4)]].map((p) => p.join(",")).join(" "), fill: color }, g);
  };
  const ellipse = (L, attrs) => NB.el("ellipse", Object.assign({ cx: X(a), cy: Y(b), rx: Math.sqrt(L) * sc, ry: Math.sqrt(L / c) * sc, fill: "none" }, attrs), gL);

  const cp = NB.el("clipPath", { id: "lg" + Math.random().toString(36).slice(2, 8) }, NB.el("defs", {}, svg));
  NB.el("rect", { x: 10, y: 12, width: pw - 10, height: H - 24 }, cp);
  const gL = NB.el("g", { "clip-path": `url(#${cp.id})` }, svg), gR = NB.el("g", {}, svg);
  const hit = NB.el("rect", { x: 10, y: 12, width: pw - 10, height: H - 24, fill: "transparent", style: { cursor: "pointer", touchAction: "none" } }, svg);
  let dragging = false;
  const move = (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const t = snap(Math.atan2(Yinv(p.y), Xinv(p.x)), 0.035);   // snap within 2 degrees
    st.t = t; sl.set(Math.round(deg(t)) % 360); draw();
  };
  hit.addEventListener("pointerdown", (e) => { dragging = true; hit.setPointerCapture(e.pointerId); move(e); });
  hit.addEventListener("pointermove", (e) => { if (dragging) move(e); });
  hit.addEventListener("pointerup", () => { dragging = false; });

  function draw() {
    NB.clear(gL); NB.clear(gR);
    NB.el("rect", { x: 10, y: 12, width: pw - 10, height: H - 24, fill: "none", stroke: NB.c.grid }, gL);
    NB.el("line", { x1: X(-3), x2: X(3), y1: Y(0), y2: Y(0), class: "nb-axis", "stroke-opacity": 0.5 }, gL);
    NB.el("line", { x1: X(0), x2: X(0), y1: Y(-3), y2: Y(3), class: "nb-axis", "stroke-opacity": 0.5 }, gL);
    [0.25, 1, 2.25, 4, 6.25, 9, 12.25, 16, 20.25].forEach((L) => ellipse(L, { stroke: NB.c.blue, "stroke-opacity": 0.3, "stroke-width": 1 }));
    NB.el("path", { d: `M${X(a) - 5},${Y(b) - 5}L${X(a) + 5},${Y(b) + 5}M${X(a) - 5},${Y(b) + 5}L${X(a) + 5},${Y(b) - 5}`, stroke: NB.c.blue, "stroke-width": 2 }, gL);
    NB.el("text", { x: X(a) + 8, y: Y(b) - 6, style: { fontSize: "12px", fill: NB.c.blue } }, gL, "min of f₀");
    NB.el("circle", { cx: X(0), cy: Y(0), r: sc, fill: "none", stroke: NB.c.green, "stroke-width": 3 }, gL);
    NB.el("text", { x: X(-0.72) - 4, y: Y(-0.72) + 16, "text-anchor": "end", style: { fontSize: "12px", fill: NB.c.green } }, gL, "h(x) = 0");
    const x1 = Math.cos(st.t), x2 = Math.sin(st.t), gf = grad(x1, x2), gh = [2 * x1, 2 * x2];
    ellipse(f0(x1, x2), { stroke: NB.c.blue, "stroke-width": 2 });   // the level set through x
    // tangent line to the constraint at x
    NB.el("line", { x1: X(x1 + 1.2 * x2), y1: Y(x2 - 1.2 * x1), x2: X(x1 - 1.2 * x2), y2: Y(x2 + 1.2 * x1), stroke: NB.c.muted, "stroke-dasharray": "4 4" }, gL);
    const s = Math.min(0.22, 0.8 / Math.hypot(...gf));   // one scale for both arrows, so their lengths compare
    arrow(gL, x1, x2, x1 + s * gh[0], x2 + s * gh[1], NB.c.green);
    arrow(gL, x1, x2, x1 + s * gf[0], x2 + s * gf[1], NB.c.blue);
    // labels just past each arrow tip, in the arrow's direction
    const tipLab = (g, color, t) => { const n = Math.hypot(...g), px = X(x1 + s * g[0]) + (16 * g[0]) / n, py = Y(x2 + s * g[1]) - (14 * g[1]) / n;
      NB.el("text", { x: px, y: py + 5, "text-anchor": "middle", style: { fontSize: "13px", fill: color, fontWeight: 600 } }, gL, t); };
    tipLab(gf, NB.c.blue, "∇f₀"); tipLab(gh, NB.c.green, "∇h");
    NB.el("circle", { cx: X(x1), cy: Y(x2), r: 6.5, fill: NB.c.pink, stroke: NB.c.bg, "stroke-width": 1.5 }, gL);

    // right: f0 along the constraint
    NB.axes(gR, Xt, Yt, { xlabel: "angle of x on the circle (degrees)", ylabel: "f₀(x)", xticks: 4, yticks: 4 });
    NB.el("path", { d: NB.line(NB.range(0, 360, 361).map((d) => [d, fth((d * Math.PI) / 180)]), Xt, Yt), fill: "none", stroke: NB.c.blue, "stroke-width": 2.2 }, gR);
    stat.forEach((p) => {
      NB.el("circle", { cx: Xt(deg(p.t)), cy: Yt(fth(p.t)), r: 4, fill: "none", stroke: NB.c.fg, "stroke-width": 1.4 }, gR);
      NB.el("text", { x: Xt(deg(p.t)), y: Yt(fth(p.t)) + (p.min ? 18 : -9), "text-anchor": "middle", class: "nb-tick" }, gR, p.min ? "min" : "max");
    });
    NB.el("circle", { cx: Xt(deg(st.t)), cy: Yt(fth(st.t)), r: 6, fill: NB.c.pink }, gR);

    const nu = -(gf[0] * gh[0] + gf[1] * gh[1]) / (gh[0] ** 2 + gh[1] ** 2);
    const cos = (gf[0] * gh[0] + gf[1] * gh[1]) / (Math.hypot(...gf) * Math.hypot(...gh));
    const ang = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
    const at = stat.find((p) => angDist(st.t, p.t) < 1e-9);
    let msg = `x = (${NB.fmt(x1, 2)}, ${NB.fmt(x2, 2)}), f₀(x) = <b>${NB.fmt(f0(x1, x2), 2)}</b>. Angle between ∇f₀ and ∇h: <b>${ang.toFixed(1)}°</b>. `;
    if (at) msg += `They are parallel, so ∇f₀ + ν∇h = 0 with <b>ν = ${NB.fmt(nu, 3)}</b> = −(∇f₀·∇h)/|∇h|². This is the constrained <b>${at.min ? "minimum" : "maximum"}</b>: ` +
      (at.min ? "moving along the circle in either direction raises f₀." : "the condition also holds where f₀ is largest, so it is necessary, not sufficient.");
    else msg += `∇f₀ has a part along the tangent of the circle (df₀/dθ = <b>${NB.fmt(dfth(st.t), 2)}</b>), so moving ${dfth(st.t) > 0 ? "clockwise" : "counterclockwise"} still lowers f₀. No ν makes ∇f₀ + ν∇h = 0 here.`;
    read.innerHTML = msg;
  }
  draw();
});

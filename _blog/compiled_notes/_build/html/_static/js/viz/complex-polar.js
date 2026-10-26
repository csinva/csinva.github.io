/* complex-polar: two complex numbers in polar form z = |z| e^{iθ} and their product (or quotient).
 * The result's magnitude (blue circle) is the product of the magnitudes and its phase (orange arc) is the
 * sum of the phases, matching the chapter's coloring of |z| (blue) and e^{iθ} (orange).
 * params: {r1: 1.5, t1: 30, r2: 1.2, t2: 60, op: "mul"} */
NB.register("complex-polar", function (el, P) {
  const st = { r1: P.r1 ?? 1.5, t1: P.t1 ?? 30, r2: P.r2 ?? 1.2, t2: P.t2 ?? 60, op: P.op || "mul" };
  const ctl = NB.controls(el);
  const deg = (v) => v.toFixed(0) + "°";
  NB.slider(ctl, { label: "|z₁|", min: 0.2, max: 2, step: 0.05, value: st.r1 }, (v) => { st.r1 = v; draw(); });
  NB.slider(ctl, { label: "θ₁", min: -180, max: 180, step: 5, value: st.t1, format: deg }, (v) => { st.t1 = v; draw(); });
  NB.slider(ctl, { label: "|z₂|", min: 0.2, max: 2, step: 0.05, value: st.r2 }, (v) => { st.r2 = v; draw(); });
  NB.slider(ctl, { label: "θ₂", min: -180, max: 180, step: 5, value: st.t2, format: deg }, (v) => { st.t2 = v; draw(); });
  NB.select(ctl, "operation", [["mul", "z₁ · z₂"], ["div", "z₁ / z₂"]], st.op, (v) => { st.op = v; draw(); });
  const W = 460, H = 460, L = 4.2;
  const svg = NB.svg(NB.el("div", { style: { maxWidth: "420px", margin: "0 auto" } }, el), W, H);
  NB.legend(el, [["z₁", NB.c.fg], ["z₂", NB.c.muted], ["result", NB.c.green], ["its magnitude", NB.c.blue, true], ["its phase", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(-L, L, 10, W - 10), Y = NB.scale(-L, L, H - 10, 10);

  function arrow(g, r, t, color, width, dash) {
    const x = r * Math.cos(t), y = r * Math.sin(t);
    NB.el("line", { x1: X(0), y1: Y(0), x2: X(x), y2: Y(y), stroke: color, "stroke-width": width, "stroke-dasharray": dash || null }, g);
    NB.el("circle", { cx: X(x), cy: Y(y), r: width + 2, fill: color }, g);
    return [X(x), Y(y)];
  }
  function draw() {
    NB.clear(svg);
    const g = NB.el("g", {}, svg);
    NB.ticks(-4, 4, 8).forEach((t) => {
      NB.el("line", { x1: X(t), x2: X(t), y1: Y(-L), y2: Y(L), class: "nb-grid" }, g);
      NB.el("line", { y1: Y(t), y2: Y(t), x1: X(-L), x2: X(L), class: "nb-grid" }, g);
      if (t) NB.el("text", { x: X(t), y: Y(0) + 14, class: "nb-tick", "text-anchor": "middle" }, g, t);
      if (t) NB.el("text", { x: X(0) - 5, y: Y(t) + 4, class: "nb-tick", "text-anchor": "end" }, g, t + "i");
    });
    NB.el("line", { x1: X(-L), x2: X(L), y1: Y(0), y2: Y(0), class: "nb-axis" }, g);
    NB.el("line", { y1: Y(-L), y2: Y(L), x1: X(0), x2: X(0), class: "nb-axis" }, g);
    NB.el("text", { x: X(L) - 4, y: Y(0) - 6, class: "nb-label", "text-anchor": "end" }, g, "real");
    NB.el("text", { x: X(0) + 6, y: Y(L) + 12, class: "nb-label" }, g, "imaginary");
    const a1 = (st.t1 * Math.PI) / 180, a2 = (st.t2 * Math.PI) / 180;
    const r = st.op === "mul" ? st.r1 * st.r2 : st.r1 / st.r2, t = st.op === "mul" ? a1 + a2 : a1 - a2;
    // magnitude of the result: circle of radius |z|
    NB.el("circle", { cx: X(0), cy: Y(0), r: X(Math.min(r, L)) - X(0), fill: "none", stroke: NB.c.blue, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, g);
    // phase of the result: arc from the real axis
    const ra = 0.55, steps = 60, pts = NB.range(0, t, steps).map((a) => `${X(ra * Math.cos(a)).toFixed(1)},${Y(ra * Math.sin(a)).toFixed(1)}`);
    NB.el("polyline", { points: pts.join(" "), fill: "none", stroke: NB.c.orange, "stroke-width": 2.5 }, g);
    const p1 = arrow(g, st.r1, a1, NB.c.fg, 2), p2 = arrow(g, st.r2, a2, NB.c.muted, 2);
    const rc = Math.min(r, L - 0.05), pr = arrow(g, rc, t, NB.c.green, 3);
    const lab = (p, s, c) => NB.el("text", { x: p[0] + 8, y: p[1] - 8, style: { fontSize: "14px", fill: c, fontWeight: 600 } }, g, s);
    lab(p1, "z₁", NB.c.fg); lab(p2, "z₂", NB.c.muted); lab(pr, st.op === "mul" ? "z₁z₂" : "z₁/z₂", NB.c.green);
    const wrap = (d) => ((((d + 180) % 360) + 360) % 360) - 180;
    const td = st.op === "mul" ? st.t1 + st.t2 : st.t1 - st.t2;
    const sym = st.op === "mul" ? ["·", "+"] : ["/", "−"];
    read.innerHTML = `magnitude <b style="color:var(--nb-c-blue)">${NB.fmt(st.r1, 2)} ${sym[0]} ${NB.fmt(st.r2, 2)} = ${NB.fmt(r, 3)}</b>, ` +
      `phase <b style="color:var(--nb-c-orange)">${st.t1}° ${sym[1]} ${st.t2}° = ${td}°</b>${wrap(td) !== td ? ` (same as ${wrap(td)}°)` : ""}` +
      (r > L ? " (result is off the plot)" : "");
  }
  draw();
});

/* backprop-graph: forward and backward pass on the computational graph f = (x + y) z, the note's backprop figure.
 * Forward values (green) are written above each wire, gradients df/d(wire) (pink) below it. Choosing a gate shows
 * its chain-rule step: the gradient arriving from the output times the gate's local derivative.
 * params: {x: -2, y: 5, z: -4, node: "mul"} */
NB.register("backprop-graph", function (el, P) {
  const st = { x: P.x ?? -2, y: P.y ?? 5, z: P.z ?? -4, node: P.node || "mul" };
  const ctl = NB.controls(el);
  ["x", "y", "z"].forEach((k) => NB.slider(ctl, { label: k, min: -5, max: 5, step: 0.5, value: st[k] }, (v) => { st[k] = v; draw(); }));
  const sel = NB.select(ctl, "chain rule at", [["mul", "× gate (f = q z)"], ["add", "+ gate (q = x + y)"]], st.node, (v) => { st.node = v; draw(); });
  const W = 640, H = 310, svg = NB.svg(el, W, H);
  NB.legend(el, [["forward value", NB.c.green], ["gradient df/d(·)", NB.c.pink]]);
  const read = NB.readout(el);
  const f2 = (v) => NB.fmt(v, 2).replace("-", "−");

  // layout (svg units): inputs on the left, + gate, × gate, output f on the right
  const nodes = { add: [250, 95], mul: [450, 165] }, R = 30;
  const wires = {   // polyline points, var name, where to write values (label x, wire y)
    x: { pts: [[52, 55], [150, 55], [nodes.add[0] - R + 2, nodes.add[1] - 10]], at: [86, 55] },
    y: { pts: [[52, 145], [150, 145], [nodes.add[0] - R + 2, nodes.add[1] + 10]], at: [86, 145] },
    z: { pts: [[52, 245], [150, 245], [nodes.mul[0] - R + 3, nodes.mul[1] + 12]], at: [86, 245] },
    q: { pts: [[nodes.add[0] + R, nodes.add[1]], [365, 95], [nodes.mul[0] - R + 4, nodes.mul[1] - 12]], at: [330, 95] },
    f: { pts: [[nodes.mul[0] + R, nodes.mul[1]], [610, 165]], at: [540, 165] },
  };

  function arrowHead(p, q, color) {
    const ang = Math.atan2(q[1] - p[1], q[0] - p[0]), L = 10;
    NB.el("polygon", { points: [q, [q[0] - L * Math.cos(ang - 0.4), q[1] - L * Math.sin(ang - 0.4)], [q[0] - L * Math.cos(ang + 0.4), q[1] - L * Math.sin(ang + 0.4)]].map((a) => a.map((v) => v.toFixed(1)).join(",")).join(" "), fill: color }, svg);
  }

  function draw() {
    const { x, y, z } = st, q = x + y, f = q * z;
    // backward: df/df = 1; × gate: df/dq = z, df/dz = q; + gate passes df/dq to x and y
    const g = { f: 1, q: z, z: q }; g.x = g.q * 1; g.y = g.q * 1;
    const val = { x, y, z, q, f };
    const active = st.node === "mul" ? { in: ["q", "z"], out: "f" } : { in: ["x", "y"], out: "q" };
    NB.clear(svg);
    Object.entries(wires).forEach(([k, w]) => {
      const on = active.in.includes(k) || active.out === k;
      NB.el("polyline", { points: w.pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: NB.c.fg, "stroke-width": on ? 3.2 : 2, "stroke-opacity": on ? 1 : 0.55, "stroke-linejoin": "round" }, svg);
      const n = w.pts.length; arrowHead(w.pts[n - 2], w.pts[n - 1], NB.c.fg);
      const [lx, ly] = w.at;
      NB.el("text", { x: lx, y: ly - 10, "font-size": 23, "font-weight": 600, fill: NB.c.green, "text-anchor": "middle" }, svg, f2(val[k]));
      NB.el("text", { x: lx, y: ly + 27, "font-size": 23, "font-weight": 600, fill: NB.c.pink, "text-anchor": "middle" }, svg, f2(g[k]));
    });
    ["x", "y", "z"].forEach((k) => NB.el("text", { x: 32, y: wires[k].pts[0][1] + 8, "font-size": 26, "font-style": "italic", "font-family": "Georgia, serif", "text-anchor": "middle" }, svg, k));
    NB.el("text", { x: 302, y: 95 - 10, "font-size": 24, "font-style": "italic", "font-family": "Georgia, serif", "text-anchor": "end" }, svg, "q");
    NB.el("text", { x: 616, y: 165 + 8, "font-size": 26, "font-style": "italic", "font-family": "Georgia, serif" }, svg, "f");
    Object.entries(nodes).forEach(([k, [cx, cy]]) => {
      const on = st.node === k;
      const c = NB.el("circle", { cx, cy, r: R, fill: on ? NB.c.yellow : NB.c.bg, "fill-opacity": on ? 0.22 : 1, stroke: NB.c.fg, "stroke-width": on ? 3.2 : 2, style: { cursor: "pointer" } }, svg);
      const t = NB.el("text", { x: cx, y: cy + 10, "font-size": 30, "text-anchor": "middle", style: { cursor: "pointer" } }, svg, k === "add" ? "+" : "×");
      [c, t].forEach((e) => e.addEventListener("click", () => { st.node = k; sel.value = k; draw(); }));
    });
    NB.el("text", { x: W / 2, y: H - 10, "font-size": 16, fill: NB.c.muted, "text-anchor": "middle" }, svg, "click a gate to see its chain-rule step");

    const G = (v) => `<b style="color:var(--nb-c-pink)">${f2(v)}</b>`, V = (v) => `<b style="color:var(--nb-c-green)">${f2(v)}</b>`;
    // finite-difference check of one gradient: nudge an input of the chosen gate by 0.01 and rerun the forward pass
    const ck = st.node === "mul" ? "z" : "x", nudged = ck === "z" ? (x + y) * (z + 0.01) : (x + 0.01 + y) * z;
    let rule;
    if (st.node === "mul") rule = `<b>× gate</b>, f = q z: local derivatives ∂f/∂q = z = ${V(z)} and ∂f/∂z = q = ${V(q)}. ` +
      `Incoming gradient df/df = ${G(1)}, so df/dq = ${G(1)} × ${V(z)} = ${G(g.q)} and df/dz = ${G(1)} × ${V(q)} = ${G(g.z)}. Each input gets the other input's value.`;
    else rule = `<b>+ gate</b>, q = x + y: local derivatives ∂q/∂x = ∂q/∂y = 1. ` +
      `Incoming gradient df/dq = ${G(g.q)} (from the × gate), so df/dx = ${G(g.q)} × 1 = ${G(g.x)} and df/dy = ${G(g.q)} × 1 = ${G(g.y)}. The gradient passes through unchanged.`;
    read.innerHTML = `f = ${V(f)}. ${rule} Check: raising ${ck} by 0.01 changes f by ${f2(nudged - f)}, which is 0.01 × df/d${ck}.`;
  }
  draw();
});

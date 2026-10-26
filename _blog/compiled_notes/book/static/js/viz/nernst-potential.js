/* nernst-potential: V_ion = 58/z * log10(X_out / X_in) in mV, as in the brain basics note.
 * A slider sets the outside:inside concentration ratio (log scale); a select picks the ion (its charge z).
 * The plot shows V against the ratio for each charge, with the current point marked.
 * params: {ion: "K", ratio: 0.1} */
NB.register("nernst-potential", function (el, P) {
  const ions = {
    K: { label: "K⁺ (z = +1)", z: 1, color: NB.c.blue },
    Na: { label: "Na⁺ (z = +1)", z: 1, color: NB.c.blue },
    Ca: { label: "Ca²⁺ (z = +2)", z: 2, color: NB.c.green },
    Cl: { label: "Cl⁻ (z = −1)", z: -1, color: NB.c.orange },
  };
  const st = { ion: P.ion || "K", lr: Math.log10(P.ratio || 0.1) };
  const ctl = NB.controls(el);
  NB.select(ctl, "ion", Object.entries(ions).map(([k, v]) => [k, v.label]), st.ion, (v) => { st.ion = v; draw(); });
  NB.slider(ctl, { label: "[X]out : [X]in", min: -2, max: 2, step: 0.05, value: st.lr, format: (v) => fmtRatio(Math.pow(10, v)) }, (v) => { st.lr = v; draw(); });
  const W = 620, H = 320, m = { l: 54, r: 14, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["z = +1", NB.c.blue], ["z = +2", NB.c.green], ["z = −1", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(-2, 2, m.l, W - m.r), Y = NB.scale(-130, 130, H - m.b, m.t);
  function fmtRatio(r) { return r >= 1 ? (+r.toPrecision(2)) + " : 1" : "1 : " + (+(1 / r).toPrecision(2)); }
  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "[X]out / [X]in (log scale)", ylabel: "equilibrium potential (mV)", xticks: 4, yticks: 6,
      xfmt: (t) => (t >= 0 ? String(Math.pow(10, t)) : "1/" + Math.pow(10, -t)) });
    NB.el("line", { x1: X(-2), x2: X(2), y1: Y(0), y2: Y(0), stroke: NB.c.muted, "stroke-width": 1 }, svg);
    const cur = ions[st.ion];
    [[1, NB.c.blue], [2, NB.c.green], [-1, NB.c.orange]].forEach(([z, col]) => {
      const pts = NB.range(-2, 2, 81).map((l) => [l, (58 / z) * l]).filter((p) => Math.abs(p[1]) <= 130);
      NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: col, "stroke-width": z === cur.z ? 3 : 1.5, "stroke-opacity": z === cur.z ? 1 : 0.3 }, svg);
    });
    const V = (58 / cur.z) * st.lr, Vc = Math.max(-130, Math.min(130, V));
    NB.el("line", { x1: X(st.lr), x2: X(st.lr), y1: Y(0), y2: Y(Vc), stroke: cur.color, "stroke-width": 1, "stroke-dasharray": "2 3" }, svg);
    NB.el("circle", { cx: X(st.lr), cy: Y(Vc), r: 6, fill: cur.color }, svg);
    const right = st.lr > 1;
    const below = (cur.z > 0) !== right;   // put the label on the side away from the line
    NB.el("text", { x: X(st.lr) + (right ? -12 : 12), y: Y(Vc) + (below ? 20 : -10), class: "nb-label", "text-anchor": right ? "end" : "start", fill: cur.color }, svg, `${NB.fmt(V, 0)} mV`);
    const zs = cur.z > 0 ? String(cur.z) : "(" + cur.z + ")";
    read.innerHTML = `V = 58/${zs} · log₁₀(${NB.fmt(Math.pow(10, st.lr), 2)}) = <b>${NB.fmt(V, 1)} mV</b>`;
  }
  draw();
});

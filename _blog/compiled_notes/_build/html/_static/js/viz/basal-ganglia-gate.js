/* basal-ganglia-gate: a toy firing-rate chain for the body movement loop of the motor note.
 * cortex (+)-> striatum (−)-> globus pallidus (−)-> VA/VL (+)-> motor cortex.
 * Globus pallidus fires tonically and holds VA/VL down; the striatum inhibits the pallidus, which releases
 * VA/VL (disinhibition). Dopamine sets the gain from cortex onto the striatum's spiny neurons.
 * Rates (0..1): striatum = min(1, 1.5 * dopamine * cortex); pallidus = 1 − striatum; VA/VL = 1 − pallidus.
 * params: {cortex: 0.8, dopamine: 1} */
NB.register("basal-ganglia-gate", function (el, P) {
  const st = { c: P.cortex === undefined ? 0.8 : P.cortex, d: P.dopamine === undefined ? 1 : P.dopamine };
  const TH = 0.5;
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "cortical command", min: 0, max: 1, step: 0.05, value: st.c }, (v) => { st.c = v; draw(); });
  NB.slider(ctl, { label: "dopamine (fraction of normal)", min: 0, max: 1, step: 0.05, value: st.d, format: (v) => Math.round(100 * v) + "%" }, (v) => { st.d = v; draw(); });
  const W = 660, H = 230;
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const boxW = 104, gap = (W - 5 * boxW) / 4, top = 40, bh = 120;
  const names = ["cortex", "striatum", "globus pallidus", "VA/VL thalamus", "motor cortex"];
  const signs = ["+", "−", "−", "+"];
  function draw() {
    NB.clear(svg);
    const s = Math.min(1, 1.5 * st.d * st.c), gp = 1 - s, v = 1 - gp, move = v > TH;
    const rates = [st.c, s, gp, v, move ? v : 0];
    const cols = [NB.c.blue, NB.c.blue, NB.c.orange, NB.c.green, NB.c.green];
    rates.forEach((r, i) => {
      const x = i * (boxW + gap);
      NB.el("rect", { x, y: top, width: boxW, height: bh, rx: 6, fill: "none", stroke: NB.c.muted, "stroke-width": 1.2 }, svg);
      NB.el("rect", { x: x + 1, y: top + bh * (1 - r), width: boxW - 2, height: bh * r, rx: 5, fill: cols[i], "fill-opacity": 0.45 }, svg);
      NB.el("text", { x: x + boxW / 2, y: top - 10, class: "nb-label", "text-anchor": "middle" }, svg, names[i]);
      NB.el("text", { x: x + boxW / 2, y: top + bh + 20, class: "nb-tick", "text-anchor": "middle" }, svg, i === 4 ? (move ? "movement" : "no movement") : "rate " + NB.fmt(r, 2));
      if (i < 4) {
        const ax = x + boxW + 4, bx = x + boxW + gap - 4, y = top + bh / 2;
        NB.el("line", { x1: ax, x2: bx - 6, y1: y, y2: y, stroke: NB.c.fg, "stroke-width": 1.6 }, svg);
        NB.el("path", { d: `M${bx},${y}L${bx - 8},${y - 5}L${bx - 8},${y + 5}Z`, fill: NB.c.fg }, svg);
        NB.el("text", { x: (ax + bx) / 2, y: y - 10, "font-size": 22, "font-weight": 700, "text-anchor": "middle", fill: signs[i] === "+" ? NB.c.green : NB.c.pink }, svg, signs[i]);
      }
    });
    // threshold line on VA/VL
    const xv = 3 * (boxW + gap);
    NB.el("line", { x1: xv - 4, x2: xv + boxW + 4, y1: top + bh * (1 - TH), y2: top + bh * (1 - TH), stroke: NB.c.fg, "stroke-dasharray": "4 3", "stroke-width": 1 }, svg);
    NB.el("text", { x: xv + boxW / 2, y: H - 8, class: "nb-tick", "text-anchor": "middle" }, svg, "dashed: threshold to act");
    read.innerHTML = move
      ? `pallidus inhibition lifted (rate ${NB.fmt(gp, 2)}), so VA/VL fires at <b>${NB.fmt(v, 2)}</b> and the action is released`
      : `pallidus still holds VA/VL down (pallidus ${NB.fmt(gp, 2)}, VA/VL ${NB.fmt(v, 2)} below threshold): <b>no movement</b>`;
  }
  draw();
});

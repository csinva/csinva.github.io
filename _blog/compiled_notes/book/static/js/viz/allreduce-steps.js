/* allreduce-steps: 4 ranks (GPUs), each holding its own vector split into 4 slices. Step through
 * ReduceScatter (rank r ends up with the sum of slice r) and AllGather (every rank gets every summed slice),
 * which together give AllReduce. Each cell shows its value and dots for the ranks whose numbers it contains.
 * params: {stage: 0} */
NB.register("allreduce-steps", function (el, P) {
  const R = 4, u = NB.rng(9);
  const v = Array.from({ length: R }, () => Array.from({ length: R }, () => 1 + Math.floor(u() * 9)));   // v[rank][slice]
  const stages = ["start: each rank's own values", "after ReduceScatter", "after AllGather (= AllReduce)"];
  let stage = P.stage || 0;
  const ctl = NB.controls(el);
  const btns = stages.map((s, k) => NB.button(ctl, ["1. start", "2. ReduceScatter", "3. AllGather"][k], () => { stage = k; draw(); }));
  const W = 640, H = 220, cw = 110, ch = 32, x0 = 110, y0 = 40;
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const rc = [NB.c.blue, NB.c.orange, NB.c.green, NB.c.purple];

  // what rank r holds for slice c at this stage: list of contributing ranks, or null
  function holds(r, c) {
    if (stage === 0) return [r];
    if (stage === 1) return c === r ? [0, 1, 2, 3] : null;
    return [0, 1, 2, 3];
  }

  function draw() {
    NB.clear(svg);
    btns.forEach((b, k) => { b.style.borderColor = k === stage ? "var(--nb-accent)" : ""; b.style.fontWeight = k === stage ? 600 : 400; });
    for (let r = 0; r < R; r++) NB.el("text", { x: x0 + r * (cw + 12) + cw / 2, y: y0 - 12, "text-anchor": "middle", style: { fontSize: "13px", fontWeight: 600, fill: rc[r] } }, svg, `rank ${r}`);
    for (let c = 0; c < R; c++) {
      NB.el("text", { x: x0 - 12, y: y0 + c * (ch + 8) + ch / 2 + 4, class: "nb-tick", "text-anchor": "end" }, svg, `slice ${c}`);
      for (let r = 0; r < R; r++) {
        const who = holds(r, c), x = x0 + r * (cw + 12), y = y0 + c * (ch + 8);
        NB.el("rect", { x, y, width: cw, height: ch, rx: 5, fill: who ? "var(--nb-panel)" : "none", stroke: who ? "var(--nb-border)" : "var(--nb-grid)", "stroke-dasharray": who ? "" : "4 3" }, svg);
        if (!who) continue;
        const val = who.reduce((s, k) => s + v[k][c], 0);
        NB.el("text", { x: x + 12, y: y + ch / 2 + 5, style: { fontSize: "14px", fontWeight: 600 } }, svg, val);
        who.forEach((k, i) => NB.el("circle", { cx: x + cw - 14 - (who.length - 1 - i) * 12, cy: y + ch / 2, r: 4.5, fill: rc[k] }, svg));
      }
    }
    const per = stage === 0 ? "its own 4 slices" : stage === 1 ? "1 summed slice (a quarter of the vector)" : "all 4 summed slices";
    read.innerHTML = `<b>${stages[stage]}</b>: each rank holds ${per}. Dots show whose numbers are added into a cell.`;
  }
  draw();
});

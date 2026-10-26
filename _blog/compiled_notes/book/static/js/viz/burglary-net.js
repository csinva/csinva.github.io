/* burglary-net: exact inference by enumeration in the burglary-alarm Bayesian network (CPTs from the chapter's figure).
 * Click a node to cycle it through unobserved -> observed true -> observed false. Every unobserved node shows its
 * posterior P(node = true | evidence), summed over all 2^5 joint assignments that agree with the evidence.
 * params: {evidence: {J: 1, M: 1}} */
NB.register("burglary-net", function (el, P) {
  const names = ["B", "E", "A", "J", "M"];
  const label = { B: "Burglary", E: "Earthquake", A: "Alarm", J: "JohnCalls", M: "MaryCalls" };
  const pA = { "11": 0.95, "10": 0.94, "01": 0.29, "00": 0.001 };
  const joint = (s) => {                                   // P(B, E, A, J, M) = P(B) P(E) P(A|B,E) P(J|A) P(M|A)
    const [b, e, a, j, m] = s, pa = pA["" + b + e], pj = a ? 0.9 : 0.05, pm = a ? 0.7 : 0.01;
    return (b ? 0.001 : 0.999) * (e ? 0.002 : 0.998) * (a ? pa : 1 - pa) * (j ? pj : 1 - pj) * (m ? pm : 1 - pm);
  };
  const states = Array.from({ length: 32 }, (_, k) => names.map((_, i) => (k >> (4 - i)) & 1));
  const ev = Object.assign({}, P.evidence || { J: 1, M: 1 });
  const posterior = () => {
    let Z = 0; const num = names.map(() => 0);
    states.forEach((s) => {
      if (names.some((nm, i) => nm in ev && ev[nm] !== s[i])) return;
      const p = joint(s); Z += p; s.forEach((v, i) => (num[i] += v * p));
    });
    return { Z, post: num.map((v) => v / Z) };
  };
  const ctl = NB.controls(el);
  const presets = [["no evidence", {}], ["John and Mary call", { J: 1, M: 1 }], ["alarm rings", { A: 1 }], ["alarm rings, earthquake", { A: 1, E: 1 }]];
  presets.forEach(([t, e]) => NB.button(ctl, t, () => { Object.keys(ev).forEach((k) => delete ev[k]); Object.assign(ev, e); draw(); }));
  const W = 640, H = 300, bw = 170, bh = 64;
  const pos = { B: [140, 50], E: [500, 50], A: [320, 150], J: [140, 250], M: [500, 250] };
  const edges = [["B", "A"], ["E", "A"], ["A", "J"], ["A", "M"]];
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  function draw() {
    NB.clear(svg);
    const defs = NB.el("defs", {}, svg), mk = NB.el("marker", { id: "bn-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
    NB.el("path", { d: "M0,0L10,5L0,10Z", fill: NB.c.muted }, mk);
    edges.forEach(([a, b]) => {
      const [x1, y1] = pos[a], [x2, y2] = pos[b], dx = x2 - x1, dy = y2 - y1;
      const t0 = Math.min(Math.abs(bw / 2 / dx || 1e9), Math.abs(bh / 2 / dy || 1e9)), t1 = 1 - t0;
      NB.el("line", { x1: x1 + dx * t0, y1: y1 + dy * t0, x2: x1 + dx * t1, y2: y1 + dy * t1, stroke: NB.c.muted, "stroke-width": 1.5, "marker-end": "url(#bn-arrow)" }, svg);
    });
    const { post } = posterior();
    names.forEach((nm, i) => {
      const [cx, cy] = pos[nm], g = NB.el("g", { style: { cursor: "pointer" } }, svg), obs = nm in ev;
      NB.el("rect", { x: cx - bw / 2, y: cy - bh / 2, width: bw, height: bh, rx: 9, fill: obs ? (ev[nm] ? NB.c.green : NB.c.pink) : NB.c.bg, "fill-opacity": obs ? 0.2 : 1, stroke: obs ? (ev[nm] ? NB.c.green : NB.c.pink) : NB.c.muted, "stroke-width": obs ? 2 : 1.2 }, g);
      NB.el("text", { x: cx, y: cy - 12, "text-anchor": "middle", style: { fontWeight: 600, fontSize: "14px" } }, g, label[nm]);
      if (obs) NB.el("text", { x: cx, y: cy + 16, "text-anchor": "middle", class: "nb-label" }, g, `observed ${ev[nm] ? "true" : "false"}`);
      else {
        const x0 = cx - bw / 2 + 12, w = bw - 72;
        NB.el("rect", { x: x0, y: cy + 4, width: w, height: 12, rx: 2, fill: NB.c.grid }, g);
        NB.el("rect", { x: x0, y: cy + 4, width: Math.max(1, w * post[i]), height: 12, rx: 2, fill: NB.c.blue }, g);
        NB.el("text", { x: x0 + w + 6, y: cy + 15, class: "nb-tick", style: { fill: "var(--nb-fg)" } }, g, NB.fmt(post[i], 3));
      }
      g.addEventListener("click", () => { if (!obs) ev[nm] = 1; else if (ev[nm]) ev[nm] = 0; else delete ev[nm]; draw(); });
    });
    const evs = Object.keys(ev).map((k) => `${k.toLowerCase()}${ev[k] ? "" : "=false"}`).join(", ");
    read.innerHTML = "B" in ev ? `evidence: ${evs}` : `P(Burglary | ${evs || "no evidence"}) = <b>${NB.fmt(post[0], 4)}</b>`;
  }
  draw();
});

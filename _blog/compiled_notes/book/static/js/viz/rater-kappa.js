/* rater-kappa: Cohen's kappa for two pathologists sorting slides into two grades.
 * Both raters call a fraction p of slides "pattern 4" (same marginals) and agree on a fraction p_o of slides.
 * The 2x2 table is then a = p - b (both 4), d = 1 - p - b (both 3), b = c = (1 - p_o)/2 (disagreements),
 * which needs p_o >= 1 - 2 min(p, 1 - p). Chance agreement p_e = p^2 + (1 - p)^2, kappa = (p_o - p_e)/(1 - p_e).
 * params: {po: 0.85, p: 0.5} */
NB.register("rater-kappa", function (el, P) {
  const st = { po: P.po ?? 0.85, p: P.p ?? 0.5 };
  const pe = (p) => p * p + (1 - p) * (1 - p);
  const kap = (po, p) => (po - pe(p)) / (1 - pe(p));
  const minPo = (p) => 1 - 2 * Math.min(p, 1 - p);

  const ctl = NB.controls(el);
  const sPo = NB.slider(ctl, { label: "raw agreement p_o", min: 0.5, max: 1, step: 0.01, value: st.po, format: (v) => Math.round(v * 100) + "%" }, (v) => { st.po = v; draw(); });
  NB.slider(ctl, { label: "slides called pattern 4", min: 0.05, max: 0.95, step: 0.01, value: st.p, format: (v) => Math.round(v * 100) + "%" }, (v) => { st.p = v; draw(); });
  const W = 660, H = 300, svg = NB.svg(el, W, H);
  NB.legend(el, [["both say pattern 3", NB.c.blue], ["both say pattern 4", NB.c.orange], ["disagree", NB.c.pink], ["κ at this mix", NB.c.green], ["κ at a 50/50 mix", NB.c.muted, true]]);
  const read = NB.readout(el);

  function draw() {
    NB.clear(svg);
    const p = st.p;
    if (st.po < minPo(p) - 1e-9) { st.po = Math.ceil(minPo(p) * 100 - 1e-6) / 100; sPo.set(st.po); }   // too few disagreements possible
    const po = st.po, b = (1 - po) / 2, a = p - b, d = 1 - p - b;
    // left: 100 slides as a 10 x 10 grid, colored by outcome
    const nb2 = Math.round(200 * b), nd = Math.max(0, Math.round(100 * d)), na = Math.max(0, 100 - nd - nb2);
    const kinds = [].concat(Array(nd).fill(NB.c.blue), Array(na).fill(NB.c.orange), Array(nb2).fill(NB.c.pink));
    const G = 22, X0 = 18, Y0 = 34;
    NB.el("text", { x: X0 + 5 * G, y: Y0 - 12, class: "nb-tick", "text-anchor": "middle" }, svg, "100 slides, two pathologists");
    kinds.forEach((c, i) => NB.el("rect", { x: X0 + (i % 10) * G, y: Y0 + Math.floor(i / 10) * G, width: G - 3, height: G - 3, rx: 3, fill: c, "fill-opacity": 0.85 }, svg));
    NB.el("text", { x: X0 + 5 * G, y: Y0 + 10 * G + 18, class: "nb-tick", "text-anchor": "middle" }, svg, `${na} + ${nd} agree, ${nb2} disagree`);
    // right: kappa against raw agreement for this mix and for 50/50
    const m = { l: 300, r: 14, t: 16, b: 44 };
    const X = NB.scale(0.5, 1, m.l, W - m.r), Y = NB.scale(-0.1, 1, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "raw agreement p_o", ylabel: "Cohen's κ", xticks: 5, yticks: 5, xfmt: (t) => Math.round(t * 100) + "%" });
    NB.el("rect", { x: X(0.5), y: Y(0.45), width: X(1) - X(0.5), height: Y(0.15) - Y(0.45), fill: NB.c.yellow, "fill-opacity": 0.18 }, svg);
    NB.el("text", { x: X(0.5) + 6, y: Y(0.45) + 14, class: "nb-tick" }, svg, "κ of the Gleason benchmarks (0.15–0.45)");
    NB.el("line", { x1: X(0.5), x2: X(1), y1: Y(0), y2: Y(0), stroke: NB.c.muted }, svg);
    const xs = NB.range(0.5, 1, 101);
    NB.el("path", { d: NB.line(xs.map((x) => [x, kap(x, 0.5)]), X, Y), fill: "none", stroke: NB.c.muted, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    NB.el("path", { d: NB.line(xs.filter((x) => x >= minPo(p) - 1e-9).map((x) => [x, Math.max(-0.1, kap(x, p))]), X, Y), fill: "none", stroke: NB.c.green, "stroke-width": 2.5 }, svg);
    NB.el("line", { x1: X(pe(p)), x2: X(pe(p)), y1: Y(-0.1), y2: Y(1), stroke: NB.c.green, "stroke-dasharray": "2 3" }, svg);   // kappa = 0 here
    NB.el("circle", { cx: X(po), cy: Y(Math.max(-0.1, kap(po, p))), r: 6, fill: NB.c.green }, svg);
    read.innerHTML = `observed agreement p<sub>o</sub> = <b>${Math.round(po * 100)}%</b>; agreement by chance alone p<sub>e</sub> = p² + (1−p)² = <b>${(pe(p) * 100).toFixed(0)}%</b>; ` +
      `κ = (p<sub>o</sub> − p<sub>e</sub>)/(1 − p<sub>e</sub>) = <b>${kap(po, p).toFixed(2)}</b>`;
  }
  draw();
});

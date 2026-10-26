/* base-rate: base rate neglect as 1000 people. Some are librarians, the rest salespeople; shy people are
 * filled dots. P(librarian | shy) = P(shy | lib) P(lib) / [P(shy | lib) P(lib) + P(shy | sales) P(sales)].
 * params: {libShare: 0.05, shyLib: 0.8, shySales: 0.2} */
NB.register("base-rate", function (el, P) {
  const st = { b: P.libShare ?? 0.05, pl: P.shyLib ?? 0.8, ps: P.shySales ?? 0.2 };
  const ctl = NB.controls(el);
  const pct = (v) => `${Math.round(100 * v)}%`;
  NB.slider(ctl, { label: "librarians among the 1000 (base rate)", min: 0.01, max: 0.5, step: 0.01, value: st.b, format: pct }, (v) => { st.b = v; draw(); });
  NB.slider(ctl, { label: "P(shy | librarian)", min: 0.05, max: 1, step: 0.05, value: st.pl, format: pct }, (v) => { st.pl = v; draw(); });
  NB.slider(ctl, { label: "P(shy | salesperson)", min: 0, max: 1, step: 0.05, value: st.ps, format: pct }, (v) => { st.ps = v; draw(); });
  const NX = 40, NY = 25, cs = 15, svg = NB.svg(el, NX * cs + 4, NY * cs + 4);
  NB.legend(el, [["librarian", NB.c.purple], ["salesperson", NB.c.orange], ["filled = shy, ring = not shy", NB.c.muted, true]]);
  const read = NB.readout(el);
  function draw() {
    NB.clear(svg);
    const nL = Math.round(1000 * st.b), nS = 1000 - nL, shyL = Math.round(nL * st.pl), shyS = Math.round(nS * st.ps);
    for (let i = 0; i < 1000; i++) {
      const col = Math.floor(i / NY), r = i % NY; // fill column by column so librarians form a block on the left
      const lib = i < nL, shy = lib ? i < shyL : i - nL < shyS;
      const c = lib ? NB.c.purple : NB.c.orange;
      NB.el("circle", { cx: 2 + col * cs + cs / 2, cy: 2 + r * cs + cs / 2, r: cs * 0.34, fill: shy ? c : "none", stroke: c, "stroke-width": 1.3, "stroke-opacity": shy ? 1 : 0.55 }, svg);
    }
    const post = shyL + shyS ? shyL / (shyL + shyS) : 0;
    const exact = (st.pl * st.b) / (st.pl * st.b + st.ps * (1 - st.b) || 1);
    read.innerHTML = `Shy people: <b>${shyL}</b> librarians and <b>${shyS}</b> salespeople, so P(librarian | shy) = ${shyL} / ${shyL + shyS} ≈ <b>${pct(post)}</b> (Bayes' rule gives ${NB.fmt(100 * exact, 1)}%). ` +
      (st.ps > 0 && st.pl > st.ps && post < 0.5
        ? `Shyness is ${NB.fmt(st.pl / st.ps, 1)}× as common among librarians, yet a shy person is still more likely a salesperson.`
        : ``);
  }
  draw();
});

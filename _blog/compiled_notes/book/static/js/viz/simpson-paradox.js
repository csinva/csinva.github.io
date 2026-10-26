/* simpson-paradox: Simpson's paradox with a binary treatment (exercises or not), outcome cholesterol and
 * confounder age (young / old), as in the note's example from Pearl. Within each age group exercise lowers
 * cholesterol by 10 (illustrative units); old people have cholesterol 40 higher. The slider moves old people
 * into the exercise group: the share of old people is 0.5 + 0.45 s among exercisers and 0.5 - 0.45 s among
 * the rest, so the pooled difference is -10 + 40 * 0.9 s, which turns positive once s > 0.28.
 * Dots are one sample of 60 people per group with exact cell counts and noise centered within each cell, so
 * the means in the readout are exactly the ones the formula gives (up to rounding the counts).
 * params: {imbalance: 0.6, view: "aggregated" | "stratified", n: 60} */
NB.register("simpson-paradox", function (el, P) {
  const n = P.n || 60, eff = -10, ageGap = 40, base = 185;
  const st = { s: P.imbalance ?? 0.6, view: P.view || "aggregated" };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "share old: exercisers vs not", min: 0, max: 1, step: 0.05, value: st.s, format: (v) => `${Math.round(Math.round((0.5 + 0.45 * v) * n) / n * 100)}% vs ${Math.round(Math.round((0.5 - 0.45 * v) * n) / n * 100)}%` }, (v) => { st.s = v; draw(); });
  NB.select(ctl, "view", [["aggregated", "aggregated (ignore age)"], ["stratified", "stratified by age"]], st.view, (v) => { st.view = v; draw(); });
  const W = 640, H = 330, m = { l: 58, r: 16, t: 14, b: 40 };
  const svg = NB.svg(el, W, H);
  const leg = NB.el("div", {}, el);
  const read = NB.readout(el);
  const Y = NB.scale(150, 250, H - m.b, m.t);
  const xc = [m.l + (W - m.l - m.r) * 0.27, m.l + (W - m.l - m.r) * 0.73];   // centers of the two treatment columns
  const col = { young: NB.c.green, old: NB.c.purple };
  // fixed noise and jitter per (group, slot) so dots don't jump around when the slider moves
  const u = NB.rng(11), noise = [0, 1].map(() => Array.from({ length: n }, () => [u.normal() * 9, (u() - 0.5) * 120]));

  function sample() {
    const people = [];
    [0, 1].forEach((t) => {
      const pOld = t ? 0.5 + 0.45 * st.s : 0.5 - 0.45 * st.s, nOld = Math.round(pOld * n);
      // center the noise within each (group, age) cell so the cell means are exact
      ["old", "young"].forEach((age) => {
        const idx = age === "old" ? [...Array(nOld).keys()] : [...Array(n - nOld).keys()].map((i) => i + nOld);
        const mu = base + (age === "old" ? ageGap : 0) + (t ? eff : 0);
        const mean = idx.length ? idx.reduce((s, i) => s + noise[t][i][0], 0) / idx.length : 0;
        idx.forEach((i) => people.push({ t, age, y: mu + noise[t][i][0] - mean, jx: noise[t][i][1] }));
      });
    });
    return people;
  }
  const avg = (a) => a.reduce((s, p) => s + p.y, 0) / a.length;

  function draw() {
    NB.clear(svg); NB.clear(leg);
    const people = sample(), strat = st.view === "stratified";
    NB.axes(svg, NB.scale(0, 1, m.l, W - m.r), Y, { xticks: 1, yticks: 5, xfmt: () => "", ylabel: "cholesterol" });
    ["does not exercise (T = 0)", "exercises (T = 1)"].forEach((s, t) => NB.el("text", { x: xc[t], y: H - m.b + 18, class: "nb-label", "text-anchor": "middle" }, svg, s));
    people.forEach((p) => NB.el("circle", { cx: xc[p.t] + p.jx, cy: Y(p.y), r: 3.6, fill: strat ? col[p.age] : NB.c.muted, "fill-opacity": strat ? 0.6 : 0.45 }, svg));
    const meanLine = (a, b, color, label, dy) => {
      NB.el("line", { x1: xc[0], y1: Y(a), x2: xc[1], y2: Y(b), stroke: color, "stroke-width": 3 }, svg);
      [a, b].forEach((v, t) => NB.el("rect", { x: xc[t] - 70, y: Y(v) - 1.5, width: 140, height: 3, fill: color }, svg));
      NB.el("text", { x: (xc[0] + xc[1]) / 2, y: Y((a + b) / 2) + dy, "text-anchor": "middle", style: { fontSize: "13px", fill: color, fontWeight: 600 } }, svg, label);
    };
    const g = (t, age) => people.filter((p) => p.t === t && (!age || p.age === age));
    const all0 = avg(g(0)), all1 = avg(g(1)), agg = all1 - all0;
    const shareOld = [0, 1].map((t) => g(t, "old").length / n);
    const within = { young: avg(g(1, "young")) - avg(g(0, "young")), old: avg(g(1, "old")) - avg(g(0, "old")) };
    const pOld = (g(0, "old").length + g(1, "old").length) / (2 * n);
    const adjusted = pOld * within.old + (1 - pOld) * within.young;   // back-door adjustment: weight by P(age)
    const sgn = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + NB.fmt(Math.abs(v), 1);
    if (strat) {
      meanLine(avg(g(0, "old")), avg(g(1, "old")), col.old, `old: ${sgn(within.old)}`, -10);
      meanLine(avg(g(0, "young")), avg(g(1, "young")), col.young, `young: ${sgn(within.young)}`, 20);
      NB.legend(leg, [["old", col.old], ["young", col.young]]);
    } else {
      meanLine(all0, all1, NB.c.pink, `all: ${sgn(agg)}`, -14);
      NB.legend(leg, [["everyone (age not shown)", NB.c.muted], ["group means", NB.c.pink]]);
    }
    const pct = (v) => Math.round(v * 100) + "%";
    read.innerHTML = `Old people are <b>${pct(shareOld[1])}</b> of exercisers and <b>${pct(shareOld[0])}</b> of non-exercisers. ` +
      `Aggregated, exercisers' cholesterol is <b>${sgn(agg)}</b> compared with non-exercisers` +
      (agg > 0.05 ? " (exercise looks harmful)" : agg < -0.05 ? " (exercise looks helpful)" : "") + `. ` +
      `Within each age group it is <b>${sgn(within.young)}</b> (young) and <b>${sgn(within.old)}</b> (old), ` +
      `and averaging these over the age distribution gives <b>${sgn(adjusted)}</b>.`;
  }
  draw();
});

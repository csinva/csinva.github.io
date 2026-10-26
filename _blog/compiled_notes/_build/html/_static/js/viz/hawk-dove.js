/* hawk-dove: Maynard Smith's game. Two animals contest a resource of value V. Hawk vs hawk: each gets
 * (V - C)/2 on average (they fight, the loser pays cost C). Hawk vs dove: hawk takes V, dove gets 0.
 * Dove vs dove: they share, V/2 each. Left: average payoff of each strategy against a population with hawk
 * fraction p. Right: replicator dynamics dp/dt = p(1 - p)(payoff_H - payoff_D) from a starting fraction.
 * params: {V: 2, C: 5, p0: 0.05} */
NB.register("hawk-dove", function (el, P) {
  const st = { V: P.V ?? 2, C: P.C ?? 5, p0: P.p0 ?? 0.05 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "resource value V", min: 0.5, max: 10, step: 0.5, value: st.V }, (v) => { st.V = v; draw(); });
  NB.slider(ctl, { label: "cost of losing a fight C", min: 0.5, max: 10, step: 0.5, value: st.C }, (v) => { st.C = v; draw(); });
  NB.slider(ctl, { label: "starting hawk fraction", min: 0.01, max: 0.99, step: 0.01, value: st.p0 }, (v) => { st.p0 = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 340, H = 270, m = { l: 46, r: 10, t: 12, b: 38 };
  const s1 = NB.svg(row, W, H), s2 = NB.svg(row, W, H);
  NB.legend(el, [["hawk", NB.c.orange], ["dove", NB.c.blue], ["hawk fraction over time", NB.c.green]]);
  const read = NB.readout(el);
  const pay = (p) => [p * (st.V - st.C) / 2 + (1 - p) * st.V, (1 - p) * st.V / 2];
  function draw() {
    NB.clear(s1); NB.clear(s2);
    const ps = NB.range(0, 1, 101), H_ = ps.map((p) => [p, pay(p)[0]]), D_ = ps.map((p) => [p, pay(p)[1]]);
    const lo = Math.min(0, ...H_.map((d) => d[1])), hi = Math.max(...H_.map((d) => d[1]), ...D_.map((d) => d[1]));
    const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo), H - m.b, m.t);
    NB.axes(s1, X, Y, { xlabel: "fraction of hawks p", ylabel: "average payoff", xticks: 5, yticks: 5 });
    NB.el("path", { d: NB.line(H_, X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2.4 }, s1);
    NB.el("path", { d: NB.line(D_, X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.4 }, s1);
    const pstar = st.V < st.C ? st.V / st.C : 1;
    NB.el("line", { x1: X(pstar), x2: X(pstar), y1: Y.range[0], y2: Y.range[1], stroke: NB.c.green, "stroke-dasharray": "4 4", "stroke-width": 1.5 }, s1);
    NB.el("text", { x: Math.min(X(pstar) + 4, W - m.r - 4), y: m.t + 12, "text-anchor": X(pstar) > W - 80 ? "end" : "start", class: "nb-tick", style: { fill: NB.c.green } }, s1, `p* = ${NB.fmt(pstar, 2)}`);
    // replicator dynamics (small Euler steps)
    const Tn = 60, dt = 0.02, traj = [[0, st.p0]];
    let p = st.p0;
    for (let t = dt; t <= Tn + 1e-9; t += dt) { const [h, d] = pay(p); p = Math.min(1, Math.max(0, p + dt * p * (1 - p) * (h - d))); if (Math.abs(t - Math.round(t)) < dt / 2) traj.push([Math.round(t), p]); }
    const X2 = NB.scale(0, Tn, m.l, W - m.r), Y2 = NB.scale(0, 1, H - m.b, m.t);
    NB.axes(s2, X2, Y2, { xlabel: "time", ylabel: "fraction of hawks", xticks: 4, yticks: 5 });
    NB.el("line", { x1: X2(0), x2: X2(Tn), y1: Y2(pstar), y2: Y2(pstar), stroke: NB.c.green, "stroke-dasharray": "4 4", "stroke-width": 1.2 }, s2);
    NB.el("path", { d: NB.line(traj, X2, Y2), fill: "none", stroke: NB.c.green, "stroke-width": 2.6 }, s2);
    read.innerHTML = st.V < st.C
      ? `V < C: hawks do better when rare and worse when common, so the population settles at a mix with <b>${NB.fmt(100 * pstar, 0)}%</b> hawks (p* = V/C). After 60 time units: ${NB.fmt(100 * p, 0)}%.`
      : `V ≥ C: a fight is worth its cost, hawks always do at least as well as doves, and the population drifts to all hawks (now ${NB.fmt(100 * p, 0)}%).`;
  }
  draw();
});

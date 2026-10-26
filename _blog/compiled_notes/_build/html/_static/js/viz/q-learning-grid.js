/* q-learning-grid: tabular Q-learning with epsilon-greedy exploration on Russell & Norvig's 4x3 grid world
 * (same world as value-iteration: R(s) = -0.04, slip 0.2 to each side split evenly, gamma = 1, exits +1 / -1).
 * Every episode starts bottom left and stops at an exit or after 200 steps. Update
 *   Q(s,a) += alpha [R(s) + gamma max_a' Q(s',a') - Q(s,a)],  with U(exit) = its reward,
 * alpha = 60 / (59 + n(s,a)) as in R&N. Greedy ties are broken at random. The display shows max_a Q(s,a) and the
 * greedy action; an arrow is green when it is an optimal action (from value iteration), pink otherwise.
 * Readout: average reward per episode over the last 100 learning episodes, and the exact utility of the
 * greedy policy from the start state (policy evaluation), against the optimum.
 * params: {episodes: 500, eps: 0.1, seed: 1} */
NB.register("q-learning-grid", function (el, P) {
  const st = { E: P.episodes ?? 500, eps: P.eps ?? 0.1, seed: P.seed ?? 1 };
  const R = -0.04, G = 1, SLIP = 0.2, NX = 4, NY = 3, WALL = "1,1", TERM = { "3,2": 1, "3,1": -1 };
  const ACTS = ["up", "down", "left", "right"];
  const D = { up: [0, 1], down: [0, -1], left: [-1, 0], right: [1, 0] };
  const SIDE = { up: ["left", "right"], down: ["left", "right"], left: ["up", "down"], right: ["up", "down"] };
  const key = (s) => s[0] + "," + s[1];
  const ok = (x, y) => x >= 0 && x < NX && y >= 0 && y < NY && key([x, y]) !== WALL;
  const mv = (s, a) => { const q = [s[0] + D[a][0], s[1] + D[a][1]]; return ok(q[0], q[1]) ? q : s; };
  const S = [];
  for (let x = 0; x < NX; x++) for (let y = 0; y < NY; y++) if (key([x, y]) !== WALL) S.push([x, y]);
  const NT = S.filter((s) => TERM[key(s)] === undefined);
  const exp = (U, s, a) => { let v = (1 - SLIP) * U[key(mv(s, a))]; SIDE[a].forEach((b) => (v += (SLIP / 2) * U[key(mv(s, b))])); return v; };

  // optimal utilities and actions (value iteration to convergence)
  let Ustar = {};
  S.forEach((s) => (Ustar[key(s)] = 0));
  for (let i = 0; i < 2000; i++) {
    const N = {};
    S.forEach((s) => { const t = TERM[key(s)]; N[key(s)] = t !== undefined ? t : R + G * Math.max(...ACTS.map((a) => exp(Ustar, s, a))); });
    Ustar = N;
  }
  const optActs = {};
  NT.forEach((s) => { const m = Math.max(...ACTS.map((a) => exp(Ustar, s, a))); optActs[key(s)] = ACTS.filter((a) => exp(Ustar, s, a) > m - 1e-9); });

  function learn() {
    const r = NB.rng(Math.imul(st.seed, 2654435761) >>> 0);
    for (let i = 0; i < 10; i++) r();
    const Q = {}, n = {};
    NT.forEach((s) => ACTS.forEach((a) => { Q[key(s) + a] = 0; n[key(s) + a] = 0; }));
    const rets = [];
    for (let ep = 0; ep < st.E; ep++) {
      let s = [0, 0], ret = 0;
      for (let t = 0; t < 200; t++) {
        let a;
        if (r() < st.eps) a = ACTS[Math.floor(r() * 4)];
        else {
          const m = Math.max(...ACTS.map((b) => Q[key(s) + b])), best = ACTS.filter((b) => Q[key(s) + b] === m);
          a = best[Math.floor(r() * best.length)];
        }
        const u = r(), took = u < 1 - SLIP ? a : u < 1 - SLIP / 2 ? SIDE[a][0] : SIDE[a][1];
        const s2 = mv(s, took), t2 = TERM[key(s2)], k = key(s) + a;
        ret += R;
        n[k]++;
        const alpha = 60 / (59 + n[k]);
        const target = t2 !== undefined ? R + G * t2 : R + G * Math.max(...ACTS.map((b) => Q[key(s2) + b]));
        if (t2 !== undefined) ret += t2;
        Q[k] += alpha * (target - Q[k]);
        s = s2;
        if (t2 !== undefined) break;
      }
      rets.push(ret);
    }
    return { Q, rets };
  }
  function greedy(Q, s) { const m = Math.max(...ACTS.map((b) => Q[key(s) + b])); return ACTS.find((b) => Q[key(s) + b] === m); }
  function policyValue(Q) {   // exact-enough policy evaluation of the greedy policy
    let U = {};
    S.forEach((s) => (U[key(s)] = 0));
    for (let i = 0; i < 3000; i++) {
      const N = {};
      S.forEach((s) => { const t = TERM[key(s)]; N[key(s)] = t !== undefined ? t : R + G * exp(U, s, greedy(Q, s)); });
      U = N;
    }
    return U["0,0"];
  }

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "episodes", min: 0, max: 2000, step: 10, value: st.E, format: (v) => v.toFixed(0) }, (v) => { st.E = v; draw(); });
  NB.slider(ctl, { label: "exploration ε", min: 0, max: 1, step: 0.01, value: st.eps }, (v) => { st.eps = v; draw(); });
  NB.button(ctl, "another run", () => { st.seed++; draw(); });
  const cs = 120, m = 6, svg = NB.svg(el, NX * cs + 2 * m, NY * cs + 2 * m);
  Object.assign(svg.style, { maxWidth: "540px", display: "block", margin: "0 auto" });
  const read = NB.readout(el);
  const arrow = { up: "↑", down: "↓", left: "←", right: "→" };

  function draw() {
    NB.clear(svg);
    const { Q, rets } = learn();
    let nOpt = 0;
    for (let x = 0; x < NX; x++) for (let y = 0; y < NY; y++) {
      const X = m + x * cs, Y = m + (NY - 1 - y) * cs, s = [x, y], k = key(s);
      if (k === WALL) { NB.el("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, fill: NB.c.fg, "fill-opacity": 0.6 }, svg); continue; }
      const t = TERM[k];
      if (t !== undefined) {
        NB.el("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, fill: t > 0 ? NB.c.blue : NB.c.pink, "fill-opacity": 0.8, stroke: NB.c.grid }, svg);
        NB.el("text", { x: X + cs / 2, y: Y + cs / 2 + 9, "text-anchor": "middle", "font-size": 26, "font-weight": 700 }, svg, t > 0 ? "+1" : "−1");
        continue;
      }
      const v = Math.max(...ACTS.map((b) => Q[k + b])), g = greedy(Q, s), good = optActs[k].includes(g);
      if (good) nOpt++;
      NB.el("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, fill: v >= 0 ? NB.c.blue : NB.c.pink, "fill-opacity": 0.06 + 0.4 * Math.min(1, Math.abs(v)), stroke: NB.c.grid }, svg);
      if (st.E > 0) NB.el("text", { x: X + cs / 2, y: Y + cs / 2 + 4, "text-anchor": "middle", style: { fill: good ? NB.c.green : NB.c.pink }, "font-size": 38, "font-weight": 700 }, svg, arrow[g]);
      NB.el("text", { x: X + cs / 2, y: Y + cs - 32, "text-anchor": "middle", "font-size": 18, "font-weight": 500 }, svg, NB.fmt(v, 3));
      NB.el("text", { x: X + cs / 2, y: Y + cs - 11, "text-anchor": "middle", "font-size": 13, fill: NB.c.muted }, svg, "U = " + NB.fmt(Ustar[k], 3));
      if (x === 0 && y === 0) NB.el("text", { x: X + 8, y: Y + 20, "font-size": 13, fill: NB.c.muted }, svg, "start");
    }
    const last = rets.slice(-100), avg = last.length ? last.reduce((a, b) => a + b, 0) / last.length : NaN;
    const pv = policyValue(Q);
    read.innerHTML = `after <b>${st.E}</b> episodes: average reward per episode while learning (last ${last.length}) <b>${last.length ? NB.fmt(avg, 2) : "–"}</b>. ` +
      `Greedy policy: optimal action in <b>${nOpt} of ${NT.length}</b> states; its expected utility from the start is <b>${pv < -3 ? "below −3 (it can loop forever)" : NB.fmt(pv, 3)}</b> (optimal ${NB.fmt(Ustar["0,0"], 3)}).`;
  }
  draw();
});

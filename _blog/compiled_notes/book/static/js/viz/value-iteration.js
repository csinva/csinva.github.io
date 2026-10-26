/* value-iteration: Bellman updates on Russell & Norvig's 4x3 grid world. Terminal states +1 and -1, one wall,
 * reward R(s) in every other state, moves go the intended way with probability 1 - slip and to each side
 * with probability slip / 2 (bumping into a wall means staying put). Shows U_k after k updates and the
 * greedy policy argmax_a sum_s' P(s'|s,a) U_k(s').
 * params: {k: 5, gamma: 1, reward: -0.04, slip: 0.2} */
NB.register("value-iteration", function (el, P) {
  const st = { k: P.k ?? 5, gamma: P.gamma ?? 1, R: P.reward ?? -0.04, slip: P.slip ?? 0.2 };
  const NX = 4, NY = 3, wall = "1,1", term = { "3,2": 1, "3,1": -1 };
  const states = [];
  for (let x = 0; x < NX; x++) for (let y = 0; y < NY; y++) if (`${x},${y}` !== wall) states.push([x, y]);
  const A = { up: [0, 1], down: [0, -1], left: [-1, 0], right: [1, 0] };
  const side = { up: ["left", "right"], down: ["left", "right"], left: ["up", "down"], right: ["up", "down"] };
  const ok = (x, y) => x >= 0 && x < NX && y >= 0 && y < NY && `${x},${y}` !== wall;
  const mv = (s, a) => { const q = [s[0] + A[a][0], s[1] + A[a][1]]; return ok(...q) ? q : s; };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "Bellman updates k", min: 0, max: 40, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; draw(); });
  NB.slider(ctl, { label: "discount γ", min: 0.5, max: 1, step: 0.01, value: st.gamma }, (v) => { st.gamma = v; draw(); });
  NB.slider(ctl, { label: "reward R(s) per step", min: -2, max: 0.1, step: 0.01, value: st.R }, (v) => { st.R = v; draw(); });
  NB.slider(ctl, { label: "slip probability", min: 0, max: 0.6, step: 0.05, value: st.slip }, (v) => { st.slip = v; draw(); });
  const cs = 120, m = 6, svg = NB.svg(el, NX * cs + 2 * m, NY * cs + 2 * m);
  Object.assign(svg.style, { maxWidth: "540px", display: "block", margin: "0 auto" });
  const read = NB.readout(el);

  const q = (U, s, a) => { const p = 1 - st.slip; let v = p * U[mv(s, a)]; side[a].forEach((b) => (v += (st.slip / 2) * U[mv(s, b)])); return v; };
  function iterate(k) {
    let U = {}; states.forEach((s) => (U[s] = 0));
    let delta = 0;
    for (let i = 0; i < k; i++) {
      const N = {};
      states.forEach((s) => {
        const t = term[s.join(",")];
        N[s] = t !== undefined ? t : st.R + st.gamma * Math.max(...Object.keys(A).map((a) => q(U, s, a)));
      });
      delta = Math.max(...states.map((s) => Math.abs(N[s] - U[s])));
      U = N;
    }
    return { U, delta };
  }
  const arrow = { up: "↑", down: "↓", left: "←", right: "→" };
  function draw() {
    NB.clear(svg);
    const { U, delta } = iterate(st.k);
    const vmax = Math.max(1, ...states.map((s) => Math.abs(U[s])));
    for (let x = 0; x < NX; x++) for (let y = 0; y < NY; y++) {
      const X = m + x * cs, Y = m + (NY - 1 - y) * cs, s = [x, y], key = `${x},${y}`;
      if (key === wall) { NB.el("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, fill: NB.c.fg, "fill-opacity": 0.6 }, svg); continue; }
      const u = U[s], t = term[key];
      NB.el("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, fill: u >= 0 ? NB.c.blue : NB.c.pink, "fill-opacity": 0.08 + 0.5 * Math.min(1, Math.abs(u) / vmax), stroke: NB.c.grid }, svg);
      NB.el("text", { x: X + cs / 2, y: Y + (t !== undefined ? cs / 2 + 8 : cs / 2 + 34), "text-anchor": "middle", "font-size": t !== undefined ? 26 : 19, "font-weight": t !== undefined ? 700 : 500 }, svg, t !== undefined ? (t > 0 ? "+1" : "−1") : NB.fmt(u, 3));
      if (t === undefined && st.k > 0) {
        const best = Object.keys(A).reduce((b, a) => (q(U, s, a) > q(U, s, b) + 1e-12 ? a : b), "up");
        NB.el("text", { x: X + cs / 2, y: Y + cs / 2 + 2, "text-anchor": "middle", style: { fill: NB.c.green }, "font-size": 38, "font-weight": 700 }, svg, arrow[best]);
      }
    }
    read.innerHTML = `after <b>${st.k}</b> updates: largest change in the last update <b>${NB.fmt(delta, 4)}</b>. ` +
      `Start state (bottom left): U = <b>${NB.fmt(U[[0, 0]], 3)}</b>. Arrows show the best action under the current utilities.`;
  }
  draw();
});

/* axon-competition: several motor axons innervate one muscle fiber and compete for a fixed supply of
 * neurotrophin. Each axon takes up a share that grows faster than its size (uptake_i ∝ a_i s_i²), and
 * shrinks at a fixed cost: ds_i/dt = T a_i s_i² / Σ_j a_j s_j² − s_i. The rich get richer, so one axon ends up
 * with the whole fiber. a_1 > 1 gives axon 1 an activity advantage (like the open eye vs. a closed one).
 * params: {n: 5, advantage: 1, seed: 3} */
NB.register("axon-competition", function (el, P) {
  const st = { n: P.n || 5, adv: P.advantage || 1, seed: P.seed || 3 };
  const T = 1, dt = 0.05, steps = 300;
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "axons at birth", min: 2, max: 8, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  NB.slider(ctl, { label: "activity advantage of axon 1", min: 1, max: 2, step: 0.05, value: st.adv, format: (v) => "×" + v.toFixed(2) }, (v) => { st.adv = v; draw(); });
  NB.button(ctl, "new random start", () => { st.seed += 1; draw(); });
  const W = 640, H = 300, m = { l: 48, r: 12, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const X = NB.scale(0, steps * dt, m.l, W - m.r), Y = NB.scale(0, 1.05, H - m.b, m.t);
  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "time", ylabel: "synaptic territory", xticks: 5, yticks: 4, xfmt: () => "" });
    const u = NB.rng(st.seed * 7919 + st.n);
    let s = Array.from({ length: st.n }, () => 0.15 + 0.1 * u());
    const a = s.map((_, i) => (i === 0 ? st.adv : 1));
    const tr = s.map((v) => [[0, v]]);
    for (let k = 1; k <= steps; k++) {
      const tot = s.reduce((acc, v, i) => acc + a[i] * v * v, 0) || 1e-12;
      s = s.map((v, i) => Math.max(0, v + dt * ((T * a[i] * v * v) / tot - v)));
      s.forEach((v, i) => tr[i].push([k * dt, v]));
    }
    tr.forEach((pts, i) => NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: i === 0 ? NB.c.orange : NB.c.blue, "stroke-width": i === 0 ? 2.8 : 1.6, "stroke-opacity": i === 0 ? 1 : 0.75 }, svg));
    const win = s.indexOf(Math.max(...s)), alive = s.filter((v) => v > 0.02).length;
    NB.el("text", { x: X(steps * dt) - 4, y: Y(s[win]) - 8, class: "nb-tick", "text-anchor": "end" }, svg, win === 0 ? "axon 1 wins" : "another axon wins");
    read.innerHTML = `axon 1 (orange) started with ${Math.round(100 * tr[0][0][1] / tr.reduce((acc, p) => acc + p[0][1], 0))}% of the territory; ` +
      `at the end <b>${alive}</b> axon${alive === 1 ? "" : "s"} still innervate${alive === 1 ? "s" : ""} the fiber, and the winner is <b>${win === 0 ? "axon 1" : "axon " + (win + 1)}</b>`;
  }
  draw();
});

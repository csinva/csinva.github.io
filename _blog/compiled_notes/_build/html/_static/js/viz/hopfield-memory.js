/* hopfield-memory: a binary Hopfield net with d = 100 units (a 10x10 image) storing P patterns with the Hebbian
 * rule W = (1/d) sum_mu xi^mu xi^mu^T (zero diagonal). Pattern 1 is a smiley, the others are random +-1 images.
 * The cue is pattern 1 with some pixels flipped; asynchronous updates s_i <- sign(sum_j W_ij s_j), one unit at a
 * time in a random order per sweep, until a full sweep changes nothing. Shows the cue, the current state, the
 * closest stored pattern, the energy E = -1/2 sum_{i!=j} W_ij s_i s_j over updates, and how often recall of
 * pattern 1 succeeds as P grows (many random pattern sets, same amount of noise).
 * params: {P: 5, noise: 0.2} */
NB.register("hopfield-memory", function (el, P0) {
  const d = 100, side = 10, PMAX = 30, TRIALS = 40;
  const smiley = [
    "..######..", ".#......#.", "#..#..#..#", "#..#..#..#", "#........#",
    "#.#....#.#", "#..####..#", "#........#", ".#......#.", "..######.."].join("").split("").map((c) => (c === "#" ? 1 : -1));
  const st = { P: P0.P ?? 5, noise: P0.noise ?? 0.2, t: 0, timer: null };

  // seeds: hash, then discard a few draws (NB.rng's first draws are correlated for nearby small seeds)
  const hash = (k) => { let h = (k + 0x9e3779b9) >>> 0; h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0; h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0; return (h ^ (h >>> 16)) >>> 0 || 1; };
  const rngFor = (k) => { const r = NB.rng(hash(k)); for (let i = 0; i < 8; i++) r(); return r; };
  const perm = (r, n) => { const a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const patternSet = (r) => [smiley].concat(Array.from({ length: PMAX - 1 }, () => Array.from({ length: d }, () => (r() < 0.5 ? 1 : -1))));

  // one run: patterns X[0..P-1], start from s, sweep in the order given by r until no unit changes.
  // Keeps the overlaps m_mu = xi^mu . s so each update costs O(P).
  function run(X, P, s0, r, record) {
    const s = s0.slice(), m = X.slice(0, P).map((x) => x.reduce((a, v, i) => a + v * s[i], 0));
    const energy = () => -0.5 * (m.reduce((a, v) => a + v * v, 0) / d - P);
    const traj = record ? { states: [s.slice()], E: [energy()] } : null;
    for (let sweep = 0; sweep < 50; sweep++) {
      let flips = 0;
      for (const i of perm(r, d)) {
        let h = -(P / d) * s[i];
        for (let mu = 0; mu < P; mu++) h += (X[mu][i] * m[mu]) / d;
        if (Math.abs(h) > 1e-12 && Math.sign(h) !== s[i]) {
          s[i] = -s[i]; flips++;
          for (let mu = 0; mu < P; mu++) m[mu] += 2 * X[mu][i] * s[i];
        }
        if (record) { traj.states.push(s.slice()); traj.E.push(energy()); }
      }
      if (!flips) break;
    }
    return record ? traj : s;
  }
  const cueOf = (x, order, k) => { const s = x.slice(); for (let j = 0; j < k; j++) s[order[j]] = -s[order[j]]; return s; };

  // the pattern set and flip order shown on screen (nested: more P adds patterns, more noise adds flips)
  const Xshow = patternSet(rngFor(1)), flipOrder = perm(rngFor(2), d);
  let traj = null, cue = null, curve = null, curveNoise = -1;

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "stored patterns P", min: 1, max: PMAX, step: 1, value: st.P, format: (v) => v.toFixed(0) }, (v) => { st.P = v; reset(true); });
  NB.slider(ctl, { label: "pixels flipped in the cue", min: 0, max: 0.45, step: 0.05, value: st.noise, format: (v) => Math.round(v * 100) + "%" }, (v) => { st.noise = v; reset(true); });
  const ctl2 = NB.controls(el);
  NB.button(ctl2, "reset to cue", () => { stop(); st.t = 0; draw(); });
  NB.button(ctl2, "one sweep (100 updates)", () => { stop(); st.t = Math.min(traj.E.length - 1, st.t + d); draw(); });
  NB.button(ctl2, "run to convergence", () => {
    stop(); if (st.t >= traj.E.length - 1) st.t = 0;
    st.timer = setInterval(() => { st.t = Math.min(traj.E.length - 1, st.t + 10); draw(); if (st.t >= traj.E.length - 1) stop(); }, 30);
  });
  const W = 640, H = 470, svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  function stop() { if (st.timer) clearInterval(st.timer); st.timer = null; }

  function reset(toEnd) {
    stop();
    const k = Math.round(st.noise * d);
    cue = cueOf(Xshow[0], flipOrder, k);
    traj = run(Xshow, st.P, cue, rngFor(3), true);
    st.t = toEnd ? traj.E.length - 1 : 0;
    if (curveNoise !== st.noise) { curve = recallCurve(k); curveNoise = st.noise; }
    draw();
  }
  // fraction of runs that end exactly on pattern 1, and the mean final overlap, for P = 1..PMAX
  function recallCurve(k) {
    const exact = new Array(PMAX).fill(0), over = new Array(PMAX).fill(0);
    for (let tr = 0; tr < TRIALS; tr++) {
      const r = rngFor(1000 + tr), X = patternSet(r), s0 = cueOf(X[0], perm(r, d), k);
      for (let P = 1; P <= PMAX; P++) {
        const s = run(X, P, s0, r, false), m = s.reduce((a, v, i) => a + v * X[0][i], 0) / d;
        exact[P - 1] += (m === 1) / TRIALS; over[P - 1] += m / TRIALS;
      }
    }
    return { exact, over };
  }

  function grid(x0, y0, s, ref, title) {
    const c = 15;
    NB.el("text", { x: x0 + (side * c) / 2, y: y0 - 8, "text-anchor": "middle", "font-size": 13 }, svg, title);
    for (let i = 0; i < d; i++) {
      const x = x0 + (i % side) * c, y = y0 + Math.floor(i / side) * c, wrong = ref && s[i] !== ref[i];
      NB.el("rect", { x: x + 0.5, y: y + 0.5, width: c - 1, height: c - 1, rx: 2,
        fill: s[i] > 0 ? NB.c.blue : "var(--nb-panel)", stroke: wrong ? NB.c.pink : "var(--nb-border)", "stroke-width": wrong ? 2 : 1 }, svg);
    }
  }

  function draw() {
    NB.clear(svg);
    const P = st.P, s = traj.states[st.t], done = st.t === traj.E.length - 1;
    const ov = Xshow.slice(0, P).map((x) => x.reduce((a, v, i) => a + v * s[i], 0) / d);
    let best = 0; ov.forEach((v, mu) => { if (Math.abs(v) > Math.abs(ov[best]) + 1e-12) best = mu; });
    const flipped = cue.reduce((a, v, i) => a + (v !== Xshow[0][i]), 0);
    const bestImg = ov[best] < 0 ? Xshow[best].map((v) => -v) : Xshow[best];
    grid(20, 30, cue, Xshow[0], `cue: ${flipped} pixels flipped`);
    grid(245, 30, s, Xshow[0], `state after ${st.t} updates`);
    grid(470, 30, bestImg, null, `closest stored: ${ov[best] < 0 ? "−" : ""}pattern ${best + 1}`);
    NB.el("text", { x: 320, y: 200, "text-anchor": "middle", class: "nb-tick" }, svg, "pink outline: pixel differs from pattern 1");

    // energy over single-unit updates (axis spans the whole run; the line stops at the current update)
    const n = traj.E.length - 1, Emin = Math.min(...traj.E), Emax = Math.max(...traj.E);
    const pad = Math.max(1, (Emax - Emin) * 0.08);
    const X1 = NB.scale(0, Math.max(n, 100), 70, 300), Y1 = NB.scale(Emin - pad, Emax + pad, 400, 245);
    NB.axes(svg, X1, Y1, { xlabel: "single-unit updates", ylabel: "energy E", xticks: 4, yticks: 4 });
    for (let k = d; k < n; k += d) NB.el("line", { x1: X1(k), x2: X1(k), y1: 245, y2: 400, stroke: NB.c.muted, "stroke-dasharray": "2 3", "stroke-opacity": 0.6 }, svg);
    NB.el("path", { d: NB.line(traj.E.slice(0, st.t + 1).map((e, i) => [i, e]), X1, Y1), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg);
    NB.el("circle", { cx: X1(st.t), cy: Y1(traj.E[st.t]), r: 4, fill: NB.c.orange }, svg);

    // recall vs number of stored patterns
    const X2 = NB.scale(1, PMAX, 400, 625), Y2 = NB.scale(0, 1, 400, 245);
    NB.axes(svg, X2, Y2, { xlabel: "stored patterns P (d = 100)", ylabel: "pattern 1 recall", xticks: 6, yticks: 4 });
    const pts = (arr) => arr.map((v, i) => [i + 1, Math.max(0, v)]);
    NB.el("path", { d: NB.line(pts(curve.over), X2, Y2), fill: "none", stroke: NB.c.purple, "stroke-width": 1.6, "stroke-dasharray": "5 3" }, svg);
    NB.el("path", { d: NB.line(pts(curve.exact), X2, Y2), fill: "none", stroke: NB.c.green, "stroke-width": 2.2 }, svg);
    NB.el("line", { x1: X2(P), x2: X2(P), y1: 245, y2: 400, stroke: NB.c.fg, "stroke-opacity": 0.35 }, svg);
    NB.el("circle", { cx: X2(P), cy: Y2(curve.exact[P - 1]), r: 4.5, fill: NB.c.green }, svg);
    NB.el("text", { x: 625, y: 238, "text-anchor": "end", class: "nb-tick" }, svg, `${TRIALS} random pattern sets per P`);
    // legends drawn in the svg so they sit under their own plot
    NB.el("line", { x1: 400, x2: 420, y1: 460, y2: 460, stroke: NB.c.green, "stroke-width": 2.2 }, svg);
    NB.el("text", { x: 425, y: 464, "font-size": 12 }, svg, "exact recall");
    NB.el("line", { x1: 515, x2: 535, y1: 460, y2: 460, stroke: NB.c.purple, "stroke-width": 1.6, "stroke-dasharray": "5 3" }, svg);
    NB.el("text", { x: 540, y: 464, "font-size": 12 }, svg, "mean overlap");
    NB.el("text", { x: 70, y: 464, "font-size": 12, fill: NB.c.muted }, svg, "dashed lines: end of each sweep");

    let txt = `P = <b>${P}</b> patterns, cue = pattern 1 with <b>${flipped}</b> of ${d} pixels flipped. `;
    if (!done) txt += `After ${st.t} updates the state matches pattern 1 on <b>${Math.round((1 + ov[0]) * 50)}</b> of ${d} pixels; E = ${NB.fmt(traj.E[st.t], 2)}.`;
    else {
      const sweeps = Math.ceil(n / d);
      const where = ov[best] === 1 ? (best === 0 ? "<b>recalled pattern 1 exactly</b>" : `<b>fell into pattern ${best + 1}</b>, a different memory`)
        : ov[best] === -1 ? `<b>fell into the inverse of pattern ${best + 1}</b>`
        : `<b>stuck in a state that is not a stored pattern</b> (closest is ${ov[best] < 0 ? "the inverse of " : ""}pattern ${best + 1}, overlap ${NB.fmt(Math.abs(ov[best]), 2)})`;
      txt += `Converged after ${sweeps} sweep${sweeps > 1 ? "s" : ""} (the last one changed nothing): ${where}; E fell from ${NB.fmt(traj.E[0], 2)} to ${NB.fmt(traj.E[n], 2)}. ` +
        `Over ${TRIALS} random pattern sets, P = ${P} recalls exactly ${Math.round(curve.exact[P - 1] * 100)}% of the time.`;
    }
    read.innerHTML = txt;
  }
  reset(true);
});

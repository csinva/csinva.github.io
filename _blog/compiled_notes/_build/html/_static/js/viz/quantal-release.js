/* quantal-release: why evoked synaptic potentials come in steps of one mini.
 * Each trial releases a whole number k of vesicles, k ~ Poisson(m) (m = mean quantal content); each vesicle adds
 * one quantum of size q = 1 with spread cv (Gaussian), and the recording adds baseline noise of sd 0.08 q.
 * Amplitude = sum of k quanta + noise ~ N(k, sqrt(k cv^2 + 0.08^2)) given k. The histogram of TRIALS seeded trials
 * is drawn against the exact mixture density sum_k Pois(k; m) N(x; k, sd_k). Failures (k = 0) have probability e^-m.
 * params: {m: 1.5, cv: 0.15} */
NB.register("quantal-release", function (el, P) {
  const TRIALS = 600, NOISE = 0.08, XMAX = 9, BW = 0.1;
  const st = { m: P.m ?? 1.5, cv: P.cv ?? 0.15 };
  const hash = (x) => { let h = (x + 0x9e3779b9) >>> 0; h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0; h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0; return (h ^ (h >>> 16)) >>> 0 || 1; };
  // fixed uniforms/normals per trial, so moving a slider changes the model, not the luck of the draw
  const r = NB.rng(hash(7)); for (let i = 0; i < 8; i++) r();
  const U = Array.from({ length: TRIALS }, () => r()), Z = Array.from({ length: TRIALS }, () => Array.from({ length: 41 }, () => r.normal()));
  const poisInv = (u, m) => { let k = 0, p = Math.exp(-m), c = p; while (u > c && k < 60) { k++; p *= m / k; c += p; } return k; };
  const pois = (k, m) => { let p = Math.exp(-m); for (let i = 1; i <= k; i++) p *= m / i; return p; };
  const dens = (x, m, cv) => { let s = 0; for (let k = 0; k <= 40; k++) { const sd = Math.sqrt(k * cv * cv + NOISE * NOISE); s += pois(k, m) * Math.exp(-((x - k) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI)); } return s; };

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "mean vesicles per spike m", min: 0.1, max: 5, step: 0.1, value: st.m, format: (v) => v.toFixed(1) }, (v) => { st.m = v; draw(); });
  NB.slider(ctl, { label: "spread of one quantum (cv)", min: 0, max: 0.4, step: 0.01, value: st.cv, format: (v) => v.toFixed(2) }, (v) => { st.cv = v; draw(); });
  const W = 660, H = 290, svg = NB.svg(el, W, H);
  NB.legend(el, [[`${TRIALS} evoked responses`, NB.c.blue], ["expected (mixture of whole quanta)", NB.c.orange], ["one mini = one vesicle", NB.c.muted, true]]);
  const read = NB.readout(el);
  const mg = { l: 52, r: 12, t: 14, b: 42 };
  const X = NB.scale(-0.5, XMAX, mg.l, W - mg.r);

  function draw() {
    NB.clear(svg);
    const amps = U.map((u, t) => {
      const k = poisInv(u, st.m);
      let a = NOISE * Z[t][0];
      for (let j = 0; j < k; j++) a += 1 + st.cv * Z[t][1 + Math.min(j, 39)];
      return { k, a };
    });
    const nb = Math.round((XMAX + 0.5) / BW), counts = new Array(nb).fill(0);
    amps.forEach(({ a }) => { const b = Math.floor((a + 0.5) / BW); if (b >= 0 && b < nb) counts[b]++; });
    const xs = NB.range(-0.5, XMAX, 600), dv = xs.map((x) => dens(x, st.m, st.cv) * TRIALS * BW);
    const ymax = Math.max(8, Math.max(...counts), Math.min(Math.max(...dv), 3 * Math.max(...counts))) * 1.08;
    const Y = NB.scale(0, ymax, H - mg.b, mg.t);
    NB.axes(svg, X, Y, { xlabel: "response amplitude (in units of one mini, q)", ylabel: "trials", xticks: 10, yticks: 5 });
    for (let k = 1; k <= 8; k++) NB.el("line", { x1: X(k), x2: X(k), y1: Y(0), y2: mg.t, stroke: NB.c.muted, "stroke-dasharray": "3 4", "stroke-opacity": 0.7 }, svg);
    counts.forEach((c, b) => { if (c) NB.el("rect", { x: X(-0.5 + b * BW) + 0.5, y: Y(c), width: Math.max(0.5, X(BW) - X(0) - 1), height: Y(0) - Y(c), fill: NB.c.blue, "fill-opacity": 0.75 }, svg); });
    NB.el("path", { d: NB.line(xs.map((x, i) => [x, Math.min(dv[i], ymax)]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg);
    const fails = amps.filter((a) => a.k === 0).length, mean = amps.reduce((s, a) => s + a.a, 0) / TRIALS;
    read.innerHTML = `failures (no vesicle released): <b>${(100 * fails / TRIALS).toFixed(1)}%</b> of trials, expected e<sup>−m</sup> = <b>${(100 * Math.exp(-st.m)).toFixed(1)}%</b>; ` +
      `mean response / mini size <b>${mean.toFixed(2)}</b> (estimates m = ${st.m.toFixed(1)})`;
  }
  draw();
});

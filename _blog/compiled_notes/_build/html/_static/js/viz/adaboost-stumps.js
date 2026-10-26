/* adaboost-stumps: AdaBoost with decision stumps on 2-d toy data, labels y = ±1.
 * Start with weights w_i = 1/n. Each round t fits the stump h_t (one feature, one threshold) with the smallest
 * weighted error ε_t, gives it the vote α_t = ½ ln((1 − ε_t)/ε_t), and reweights w_i ← w_i exp(−α_t y_i h_t(x_i))
 * (then normalizes), so the points h_t got wrong count for more in round t + 1. The ensemble is sign(Σ α_t h_t).
 * Left: points sized by the weights round T was trained on, stump h_T (dashed), the ensemble's regions.
 * Right: training error of the ensemble and weighted error ε_t of each stump.
 * params: {rounds: 5, shape: "diagonal"} */
NB.register("adaboost-stumps", function (el, P) {
  const n = 60, maxT = 40;
  function makeData(shape) {
    const r = NB.rng(3), X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const x1 = r(), x2 = r();
      const y = shape === "circle" ? ((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 < 0.32 ** 2 ? 1 : -1) : (x2 > 1.1 - x1 ? 1 : -1);
      X.push([x1, x2]); Y.push(y);
    }
    return { X, Y };
  }
  const stumpPred = (h, x) => (x[h.j] > h.t ? h.s : -h.s);
  function boost({ X, Y }) {
    let w = X.map(() => 1 / n);
    const F = X.map(() => 0), rounds = [];
    for (let t = 0; t < maxT; t++) {
      let best = null;
      for (let j = 0; j < 2; j++) {
        const v = [...new Set(X.map((x) => x[j]))].sort((a, b) => a - b);
        const ths = [v[0] - 0.01, ...v.slice(1).map((u, k) => (u + v[k]) / 2)];
        for (const th of ths) for (const s of [1, -1]) {
          let e = 0;
          for (let i = 0; i < n; i++) if ((X[i][j] > th ? s : -s) !== Y[i]) e += w[i];
          if (!best || e < best.e - 1e-12) best = { j, t: th, s, e };
        }
      }
      const a = 0.5 * Math.log((1 - best.e) / Math.max(best.e, 1e-10));
      const h = X.map((x) => stumpPred(best, x));
      h.forEach((v, i) => (F[i] += a * v));
      const err = F.filter((f, i) => Math.sign(f) !== Y[i]).length / n;
      rounds.push({ stump: best, alpha: a, w: w.slice(), err, F: F.slice() });
      w = w.map((wi, i) => wi * Math.exp(-a * Y[i] * h[i]));
      const z = w.reduce((p, q) => p + q, 0); w = w.map((wi) => wi / z);
    }
    return rounds;
  }
  const sets = { diagonal: makeData("diagonal"), circle: makeData("circle") };
  const runs = { diagonal: boost(sets.diagonal), circle: boost(sets.circle) };
  const st = { T: P.rounds ?? 5, shape: P.shape || "diagonal" };

  const ctl = NB.controls(el);
  NB.select(ctl, "data", [["diagonal", "diagonal boundary"], ["circle", "circle"]], st.shape, (v) => { st.shape = v; draw(); });
  NB.slider(ctl, { label: "rounds T", min: 1, max: maxT, step: 1, value: st.T, format: (v) => v.toFixed(0) }, (v) => { st.T = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px" } }, el);
  const S = 360, s1 = NB.svg(row, S, S), W2 = 340, H2 = 360, s2 = NB.svg(row, W2, H2);
  NB.legend(el, [["y = +1", NB.c.blue], ["y = −1", NB.c.purple], ["stump h_T", NB.c.fg, true], ["ensemble training error", NB.c.pink], ["stump's weighted error ε_t", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, 8, S - 8), Y = NB.scale(0, 1, S - 8, 8);
  const m = { l: 44, r: 10, t: 12, b: 40 };
  const Xt = NB.scale(0, maxT, m.l, W2 - m.r), Ye = NB.scale(0, 0.5, H2 - m.b, m.t);

  function draw() {
    const D = sets[st.shape], R = runs[st.shape], T = st.T, cur = R[T - 1], h = cur.stump;
    NB.clear(s1); NB.clear(s2);
    // ensemble regions: Σ α_t h_t is constant between consecutive thresholds, so color those cells exactly
    const cuts = [[0, 1], [0, 1]];
    R.slice(0, T).forEach((r) => { if (r.stump.t > 0 && r.stump.t < 1) cuts[r.stump.j].push(r.stump.t); });
    cuts.forEach((c) => c.sort((a, b) => a - b));
    const g = NB.el("g", {}, s1);
    for (let a = 0; a + 1 < cuts[0].length; a++) for (let b = 0; b + 1 < cuts[1].length; b++) {
      const c = [(cuts[0][a] + cuts[0][a + 1]) / 2, (cuts[1][b] + cuts[1][b + 1]) / 2];
      const f = R.slice(0, T).reduce((s, r) => s + r.alpha * stumpPred(r.stump, c), 0);
      NB.el("rect", { x: X(cuts[0][a]), y: Y(cuts[1][b + 1]), width: X(cuts[0][a + 1]) - X(cuts[0][a]) + 0.4, height: Y(cuts[1][b]) - Y(cuts[1][b + 1]) + 0.4, fill: f > 0 ? NB.c.blue : NB.c.purple, "fill-opacity": 0.13 }, g);
    }
    NB.el("rect", { x: 8, y: 8, width: S - 16, height: S - 16, fill: "none", stroke: NB.c.grid }, s1);
    // points: marker area proportional to the weight w_i that round T was trained on
    const order = D.X.map((_, i) => i).sort((a, b) => cur.w[b] - cur.w[a]);
    order.forEach((i) => {
      const x = D.X[i], y = D.Y[i], r = Math.min(16, 2.2 + 3.6 * Math.sqrt(cur.w[i] * n)), wrong = stumpPred(h, x) !== y;
      const a = { fill: y > 0 ? NB.c.blue : NB.c.purple, "fill-opacity": 0.75, stroke: wrong ? NB.c.fg : "none", "stroke-width": 1.4 };
      if (y > 0) NB.el("circle", Object.assign({ cx: X(x[0]), cy: Y(x[1]), r }, a), s1);
      else NB.el("rect", Object.assign({ x: X(x[0]) - r * 0.88, y: Y(x[1]) - r * 0.88, width: r * 1.76, height: r * 1.76 }, a), s1);
    });
    // current stump: a vertical or horizontal line, with the side it calls +1 marked
    if (h.t > 0 && h.t < 1) {
      const p = h.j === 0 ? { x1: X(h.t), x2: X(h.t), y1: Y(0), y2: Y(1) } : { x1: X(0), x2: X(1), y1: Y(h.t), y2: Y(h.t) };
      NB.el("line", Object.assign({ stroke: NB.c.fg, "stroke-width": 2.4, "stroke-dasharray": "7 5" }, p), s1);
      const plusHigh = h.s > 0, lbl = "h_T = +1";
      if (h.j === 0) NB.el("text", { x: X(h.t) + (plusHigh ? 6 : -6), y: 26, "font-size": 13, "text-anchor": plusHigh ? "start" : "end" }, s1, plusHigh ? lbl + " →" : "← " + lbl);
      else NB.el("text", { x: S - 14, y: Y(h.t) + (plusHigh ? -8 : 18), "font-size": 13, "text-anchor": "end" }, s1, plusHigh ? "↑ " + lbl : "↓ " + lbl);
    }

    NB.axes(s2, Xt, Ye, { xlabel: "round t", ylabel: "error", xticks: 4, yticks: 5 });
    NB.el("path", { d: NB.line(R.map((r, t) => [t + 1, r.stump.e]), Xt, Ye), fill: "none", stroke: NB.c.orange, "stroke-width": 1.6 }, s2);
    NB.el("path", { d: NB.line(R.map((r, t) => [t + 1, r.err]), Xt, Ye), fill: "none", stroke: NB.c.pink, "stroke-width": 2.4 }, s2);
    NB.el("line", { x1: Xt(T), x2: Xt(T), y1: Ye(0), y2: Ye(0.5), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, s2);
    NB.el("circle", { cx: Xt(T), cy: Ye(cur.err), r: 3.6, fill: NB.c.pink }, s2);

    const f = (v) => NB.fmt(v, 2), wmax = Math.max(...cur.w);
    const rule = h.t <= 0 || h.t >= 1 ? `a constant stump (predicts ${h.s > 0 ? "+1" : "−1"} everywhere)` : `x${h.j + 1} ${h.s > 0 ? ">" : "≤"} ${f(h.t)} → +1`;
    read.innerHTML = `round ${T}: stump h_${T} is ${rule}, weighted error ε = <b>${f(h.e)}</b>, vote α = <b>${f(cur.alpha)}</b>. ` +
      `The ensemble of ${T} stump${T > 1 ? "s" : ""} misclassifies <b>${Math.round(cur.err * n)}</b> of ${n} training points (${Math.round(cur.err * 100)}%). ` +
      `Largest point weight ${f(wmax)} (uniform is ${f(1 / n)}).`;
  }
  draw();
});

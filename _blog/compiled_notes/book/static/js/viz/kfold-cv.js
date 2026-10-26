/* kfold-cv: k-fold cross-validation for choosing a polynomial degree. n = 24 points from y = sin(2πx) + noise
 * (σ = 0.3), split at random into k equal folds. For every degree the fit is redone k times, each time on k-1
 * folds, and scored on the held-out fold; the CV estimate is the average of the k fold errors. The test error
 * of the fit on all n points is computed exactly against the true curve: σ² + mean over x of (f̂(x) - f(x))².
 * For k = n (LOOCV) the readout also gives the OLS shortcut mean((e_i / (1 - h_ii))²), which needs no refitting.
 * Polynomials use the Legendre basis on t = 2x - 1 so the normal equations stay well conditioned.
 * params: {k: 4, degree: 3, seed: 9} */
NB.register("kfold-cv", function (el, P) {
  const N = 24, SIG = 0.3, DMAX = 9, KS = [2, 3, 4, 6, 8, 12, 24];
  const st = { k: P.k || 4, d: P.degree ?? 3, seed: P.seed || 9 };
  const f = (x) => Math.sin(2 * Math.PI * x);
  const basis = (x, d) => { const t = 2 * x - 1, p = [1, t];   // Legendre P_0..P_d
    for (let j = 1; j < d; j++) p.push(((2 * j + 1) * t * p[j] - j * p[j - 1]) / (j + 1));
    return p.slice(0, d + 1); };
  const fit = (xs, ys, d) => NB.lstsq(xs.map((x) => basis(x, d)), ys, 0);
  const pred = (w, x) => basis(x, w.length - 1).reduce((s, b, j) => s + b * w[j], 0);
  const grid = NB.range(0, 1, 401);
  let data;

  function sample() {
    const r = NB.rng(9173 * st.seed + 11); for (let i = 0; i < 5; i++) r();
    const xs = Array.from({ length: N }, () => r()).sort((a, b) => a - b);
    const ys = xs.map((x) => f(x) + SIG * r.normal());
    const perm = xs.map((_, i) => i);   // random fold labels: shuffle, then deal out like cards
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const rank = []; perm.forEach((p, i) => (rank[p] = i));
    data = { xs, ys, rank };
  }
  function compute() {
    const { xs, ys, rank } = data, k = st.k, fold = rank.map((r) => r % k);
    const res = [];
    for (let d = 0; d <= DMAX; d++) {
      const wAll = fit(xs, ys, d);
      const train = xs.reduce((s, x, i) => s + (ys[i] - pred(wAll, x)) ** 2, 0) / N;
      const test = SIG * SIG + grid.reduce((s, x) => s + (pred(wAll, x) - f(x)) ** 2, 0) / grid.length;
      const folds = [], fits = [];
      for (let j = 0; j < k; j++) {
        const tr = xs.map((_, i) => i).filter((i) => fold[i] !== j), te = xs.map((_, i) => i).filter((i) => fold[i] === j);
        const w = fit(tr.map((i) => xs[i]), tr.map((i) => ys[i]), d);
        fits.push(w);
        folds.push(te.reduce((s, i) => s + (ys[i] - pred(w, xs[i])) ** 2, 0) / te.length);
      }
      const cv = folds.reduce((a, b) => a + b, 0) / k;
      // LOOCV shortcut for OLS: leverage h_ii = b_i^T (B^T B)^{-1} b_i
      const B = xs.map((x) => basis(x, d)), A = Array.from({ length: d + 1 }, (_, a) => Array.from({ length: d + 1 }, (_, c) => B.reduce((s, r) => s + r[a] * r[c], 0)));
      const short = B.reduce((s, b, i) => { const v = NB.solve(A, b), h = b.reduce((t, bj, j) => t + bj * v[j], 0);
        return s + ((ys[i] - pred(wAll, xs[i])) / (1 - h)) ** 2; }, 0) / N;
      res.push({ d, train, test, folds, cv, fits, wAll, short });
    }
    return { res, fold };
  }

  const ctl = NB.controls(el);
  NB.select(ctl, "folds k", KS.map((v) => [String(v), v === N ? "24 (LOOCV)" : String(v)]), String(st.k), (v) => { st.k = +v; draw(); });
  NB.slider(ctl, { label: "degree", min: 0, max: DMAX, step: 1, value: st.d, format: (v) => v.toFixed(0) }, (v) => { st.d = v; draw(); });
  NB.button(ctl, "new data", () => { st.seed++; sample(); draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "14px" } }, el);
  const W = 400, H = 300, m = { l: 46, r: 10, t: 12, b: 38 };
  const s1 = NB.svg(row, W, H), s2 = NB.svg(row, W, H);
  NB.legend(el, [["true curve", NB.c.fg, true], ["fits, one per left-out fold", NB.c.blue], ["training error", NB.c.muted, true],
    ["CV estimate (dots: single folds)", NB.c.blue], ["test error", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(-2, 2, H - m.b, m.t);
  const D = NB.scale(-0.4, DMAX + 0.4, m.l, W - m.r), lo = Math.log10(0.02), hi = Math.log10(20), E = NB.scale(lo, hi, H - m.b, m.t);
  const ey = (v) => E(Math.min(hi, Math.max(lo, Math.log10(v))));

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const { res, fold } = compute(), r = res[st.d];
    // left: data and the k fold fits at the chosen degree
    NB.axes(s1, X, Y, { xlabel: "x", ylabel: "y", xticks: 5, yticks: 4 });
    const clip = (v) => Math.max(-2, Math.min(2, v));
    r.fits.forEach((w) => NB.el("path", { d: NB.line(grid.filter((_, i) => i % 4 === 0).map((x) => [x, clip(pred(w, x))]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 1.2, "stroke-opacity": Math.max(0.3, 0.9 - 0.025 * st.k) }, s1));
    NB.el("path", { d: NB.line(grid.map((x) => [x, f(x)]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-width": 1.6, "stroke-dasharray": "5 4" }, s1);
    data.xs.forEach((x, i) => NB.el("circle", { cx: X(x), cy: Y(clip(data.ys[i])), r: 3.6, fill: NB.c.fg, "fill-opacity": 0.85 }, s1));
    // right: error vs degree
    NB.axes(s2, D, E, { xlabel: "polynomial degree", ylabel: "mean squared error", xticks: 9, yticks: 4, yfmt: (t) => NB.fmt(Math.pow(10, t), 2) });
    NB.el("line", { x1: D(st.d), x2: D(st.d), y1: E(lo), y2: E(hi), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, s2);
    res.forEach((q) => q.folds.forEach((v, j) => NB.el("circle", { cx: D(q.d) + (st.k > 1 ? (j / (st.k - 1) - 0.5) * 14 : 0), cy: ey(v), r: 1.9, fill: NB.c.blue, "fill-opacity": 0.35 }, s2)));
    const curve = (key, color, w, dash) => NB.el("path", { d: res.map((q, i) => (i ? "L" : "M") + D(q.d).toFixed(1) + "," + ey(q[key]).toFixed(1)).join(""), fill: "none", stroke: color, "stroke-width": w, "stroke-dasharray": dash || null }, s2);
    curve("train", NB.c.muted, 1.6, "4 3"); curve("test", NB.c.orange, 2.2); curve("cv", NB.c.blue, 2.2);
    const best = (key) => res.reduce((b, q) => (q[key] < b[key] ? q : b), res[0]).d;
    const bc = best("cv"), bt = best("test");
    NB.el("circle", { cx: D(bc), cy: ey(res[bc].cv), r: 5.5, fill: "none", stroke: NB.c.blue, "stroke-width": 2 }, s2);
    NB.el("circle", { cx: D(bt), cy: ey(res[bt].test), r: 5.5, fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, s2);
    const sd = Math.sqrt(r.folds.reduce((s, v) => s + (v - r.cv) ** 2, 0) / Math.max(1, st.k - 1));
    let txt = `degree ${st.d}: training error <b>${NB.fmt(r.train, 3)}</b> · CV estimate <b>${NB.fmt(r.cv, 3)}</b> (fold errors spread ± ${NB.fmt(sd, 2)}) · test error <b>${NB.fmt(r.test, 3)}</b>` +
      `<br>CV picks degree <b>${bc}</b>; the test error is lowest at degree <b>${bt}</b>. Each fit sees ${N - N / st.k} of the ${N} points.`;
    if (st.k === N) txt += `<br>LOOCV by refitting ${N} times: <b>${NB.fmt(r.cv, 4)}</b> · by the shortcut mean((e<sub>i</sub>/(1 − h<sub>ii</sub>))²) from one fit: <b>${NB.fmt(r.short, 4)}</b>`;
    read.innerHTML = txt;
  }
  sample(); draw();
});

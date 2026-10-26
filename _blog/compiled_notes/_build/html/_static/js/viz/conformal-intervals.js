/* conformal-intervals: split conformal prediction for 1-d regression with noise that grows with x.
 * A kernel smoother f̂ is fit on 60 training points. On n calibration points the scores are |y - f̂(x)|, and
 * q̂ is the ceil((n+1)(1-α))-th smallest score (infinite if that index is > n). The band f̂ ± q̂ is drawn over
 * fresh test points. Coverage is averaged over R calibration/test draws (same fitted f̂), and compared with the
 * naive band that uses the (1-α) quantile of the training residuals.
 * Checked against numpy with the same xorshift data (see report): coverages match to 4 decimals.
 * params: {alpha: 0.1, n: 100} */
NB.register("conformal-intervals", function (el, P) {
  const st = { alpha: P.alpha || 0.1, n: P.n || 100 };
  const NTR = 60, NCALMAX = 500, NTEST = 200, R = 200, h = 0.03;
  const mu = (x) => 0.8 * Math.sin(2 * Math.PI * x) + 0.3 * x;
  const sig = (x) => 0.05 + 0.45 * x;                    // noise sd grows with x (heteroscedastic)
  const gen = (rng, n) => Array.from({ length: n }, () => { const x = rng(); return [x, mu(x) + sig(x) * rng.normal()]; });
  const train = gen(NB.rng(7), NTR);
  const fhat = (x) => { let a = 0, b = 0; for (const [xi, yi] of train) { const w = Math.exp(-0.5 * ((x - xi) / h) ** 2); a += w * yi; b += w; } return a / b; };
  // training residual quantile (type-7, like numpy.quantile): the naive interval
  const rtrain = train.map(([x, y]) => Math.abs(y - fhat(x))).sort((a, b) => a - b);
  const quant = (s, p) => { const t = p * (s.length - 1), i = Math.floor(t), f = t - i; return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i]; };
  // R independent calibration pools and test sets; only scores are needed after this
  const rng = NB.rng(12);
  const draws = Array.from({ length: R }, () => {
    const cal = gen(rng, NCALMAX), test = gen(rng, NTEST);
    return { cal, test, sc: cal.map(([x, y]) => Math.abs(y - fhat(x))), st: test.map(([x, y]) => Math.abs(y - fhat(x))) };
  });
  const grid = NB.range(0, 1, 201).map((x) => [x, fhat(x)]);
  const qhat = (scores, n, a) => { const k = Math.ceil((n + 1) * (1 - a)); if (k > n) return { q: Infinity, k }; const s = scores.slice(0, n).sort((u, v) => u - v); return { q: s[k - 1], k }; };

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "miscoverage α", min: 0.02, max: 0.5, step: 0.01, value: st.alpha }, (v) => { st.alpha = v; draw(); });
  NB.slider(ctl, { label: "calibration size n", min: 5, max: NCALMAX, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  const W = 640, H = 300, H2 = 150, m = { l: 48, r: 14, t: 12, b: 40 };
  const svg = NB.svg(el, W, H), svg2 = NB.svg(el, W, H2);
  NB.legend(el, [["fit (on 60 training points, not shown)", NB.c.blue], ["conformal band: fit ± q̂", NB.c.blue], ["band from training residuals", NB.c.orange, true],
                 ["test point inside", NB.c.muted], ["test point outside", NB.c.pink]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(-1.8, 2, H - m.b, m.t);
  const smax = 1.2, Xs = NB.scale(0, smax, m.l, W - m.r);
  const uid = "cf" + Math.random().toString(36).slice(2, 8);
  const ord = (k) => k + ([11, 12, 13].includes(k % 100) ? "th" : ["th", "st", "nd", "rd"][k % 10] || "th");
  const pct = (v) => (100 * v).toFixed(1) + "%";

  function draw() {
    const { alpha, n } = st;
    NB.clear(svg); NB.clear(svg2);
    // average coverage over all draws
    let cov = 0, covNaive = 0;
    const qn = quant(rtrain, 1 - alpha);
    draws.forEach((d) => {
      const { q } = qhat(d.sc, n, alpha);
      cov += d.st.filter((e) => e <= q).length / NTEST;
      covNaive += d.st.filter((e) => e <= qn).length / NTEST;
    });
    cov /= R; covNaive /= R;
    const d0 = draws[0], { q, k } = qhat(d0.sc, n, alpha);

    NB.axes(svg, X, Y, { xlabel: "x", ylabel: "y", xticks: 5, yticks: 4 });
    const cp = NB.el("clipPath", { id: uid }, NB.el("defs", {}, svg));
    NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, cp);
    const g = NB.el("g", { "clip-path": `url(#${uid})` }, svg);
    const qq = isFinite(q) ? q : 10;
    const band = grid.map(([x, f]) => [x, f + qq]).concat(grid.slice().reverse().map(([x, f]) => [x, f - qq]));
    NB.el("path", { d: NB.line(band, X, Y) + "Z", fill: NB.c.blue, "fill-opacity": 0.16, stroke: "none" }, g);
    [1, -1].forEach((s) => NB.el("path", { d: NB.line(grid.map(([x, f]) => [x, f + s * qn]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 1.6, "stroke-dasharray": "6 4" }, g));
    d0.test.forEach(([x, y], i) => {
      const out = d0.st[i] > q;
      NB.el("circle", { cx: X(x), cy: Y(y), r: out ? 3.4 : 2.6, fill: out ? NB.c.pink : NB.c.muted, "fill-opacity": out ? 0.95 : 0.6 }, g);
    });
    NB.el("path", { d: NB.line(grid, X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.4 }, g);

    // calibration scores of the shown draw, with the conformal quantile and the training-residual quantile
    const mb = { t: 10, b: 36 }, nb = 24, bw = smax / nb;
    const cnt = new Array(nb).fill(0);
    d0.sc.slice(0, n).forEach((s) => cnt[Math.min(nb - 1, Math.floor(s / bw))]++);
    const Yc = NB.scale(0, Math.max(4, ...cnt), H2 - mb.b, mb.t);
    NB.axes(svg2, Xs, Yc, { xlabel: "calibration score = |y − fit(x)|", ylabel: "count", xticks: 6, yticks: 3, grid: false });
    cnt.forEach((c, i) => {
      if (!c) return;
      const above = (i + 0.5) * bw > q;
      NB.el("rect", { x: Xs(i * bw) + 1, y: Yc(c), width: Xs(bw) - Xs(0) - 2, height: Yc(0) - Yc(c), fill: above ? NB.c.pink : NB.c.blue, "fill-opacity": 0.55 }, svg2);
    });
    NB.el("line", { x1: Xs(qn), x2: Xs(qn), y1: Yc.range[0], y2: mb.t, stroke: NB.c.orange, "stroke-width": 1.6, "stroke-dasharray": "6 4" }, svg2);
    if (q <= smax) {
      NB.el("line", { x1: Xs(q), x2: Xs(q), y1: Yc.range[0], y2: mb.t, stroke: NB.c.blue, "stroke-width": 2 }, svg2);
      NB.el("text", { x: Xs(q) + (q > qn ? 6 : -6), y: mb.t + 12, class: "nb-tick", style: { fill: NB.c.blue }, "text-anchor": q > qn ? "start" : "end" }, svg2, "q̂");
    }

    const lo = 1 - alpha, hi = Math.min(1, k / (n + 1));
    const half = (a, b) => { let c = 0, t = 0; d0.test.forEach(([x], i) => { if (x >= a && x < b) { t++; if (d0.st[i] <= q) c++; } }); return c / t; };
    const qtxt = isFinite(q)
      ? `q̂ = <b>${NB.fmt(q, 3)}</b>, the ${ord(k)} smallest of ${n} calibration scores`
      : `q̂ = <b>∞</b>: ⌈(n+1)(1−α)⌉ = ${k} is more than n = ${n}, so the band is the whole line (more calibration points needed)`;
    read.innerHTML = `${qtxt}. Test coverage averaged over ${R} calibration/test draws: <b>${pct(cov)}</b> ` +
      `(guaranteed at least 1 − α = ${pct(lo)}${isFinite(q) ? `; expected value ${k}/${n + 1} = ${pct(hi)}` : ""}). ` +
      `Using the ${pct(lo)} quantile of the training residuals instead: <b>${pct(covNaive)}</b>. ` +
      (isFinite(q) ? `In the draw shown, the band covers ${pct(half(0, 0.5))} of test points with x < 0.5 but ${pct(half(0.5, 1.01))} with x > 0.5, since its width is the same everywhere.` : "");
  }
  draw();
});

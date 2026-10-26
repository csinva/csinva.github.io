/* epsilon-n: the definition of convergence. (a_n) -> a if for every eps > 0 there is an N with |a_n - a| < eps for all n >= N.
 * The plot shows a_n for n = 1..40 and the band a ± eps; the widget finds the smallest such N, which can lie past
 * the plotted range. N is not read off the picture: each sequence has a bound past which every term is inside
 * the band, and the terms before it are scanned:
 *   1/n, (-1)^n/n:     |a_n - 0| = 1/n < eps for all n > 1/eps (monotone), so N = floor(1/eps) + 1
 *   (1 + 1/n)^n -> e:  increases to e, so N is the first n with e - a_n < eps
 *   sin(n)/sqrt(n):    |a_n| <= 1/sqrt(n) < eps for n > 1/eps^2; terms up to there are scanned for the last one outside
 *   (-1)^n:            diverges; the band is drawn around 0. For eps <= 1 no N works for any candidate limit a,
 *                      since 1 and -1 both recur and are 2 apart; for eps > 1, N = 1 works with a = 0.
 * Checked against a python scan to n = 10^6 for eps in {1, 0.3, 0.1, 0.05, 0.01} (e.g. 1/n: N = 11 at eps = 0.1;
 * (1+1/n)^n: N = 13 at eps = 0.1, N = 135 at eps = 0.01; sin(n)/sqrt(n): N = 100 at eps = 0.1, N = 9974 at eps = 0.01).
 * params: {seq: "inv" | "alt" | "e" | "sin" | "pm1", eps: 0.1} */
NB.register("epsilon-n", function (el, P) {
  const SEQ = {
    inv: { label: "1/n → 0", f: (n) => 1 / n, a: 0, ylo: -0.35, yhi: 1.1, alab: "0" },
    alt: { label: "(−1)ⁿ/n → 0", f: (n) => (n % 2 ? -1 : 1) / n, a: 0, ylo: -1.1, yhi: 0.65, alab: "0" },
    e: { label: "(1 + 1/n)ⁿ → e", f: (n) => Math.pow(1 + 1 / n, n), a: Math.E, ylo: 1.9, yhi: 3.0, alab: "e" },
    sin: { label: "sin(n)/√n → 0", f: (n) => Math.sin(n) / Math.sqrt(n), a: 0, ylo: -0.65, yhi: 0.95, alab: "0" },
    pm1: { label: "(−1)ⁿ (diverges)", f: (n) => (n % 2 ? -1 : 1), a: 0, ylo: -1.6, yhi: 1.6, alab: "0", diverges: true },
  };
  const st = { seq: P.seq || "inv", eps: P.eps ?? 0.1 };
  const ctl = NB.controls(el);
  NB.select(ctl, "sequence", Object.keys(SEQ).map((k) => [k, SEQ[k].label]), st.seq, (v) => { st.seq = v; draw(); });
  NB.slider(ctl, { label: "ε", min: -2, max: 0.2, step: 0.01, value: Math.log10(st.eps), format: (v) => NB.fmt(Math.pow(10, v), 3) },
    (v) => { st.eps = Math.pow(10, v); draw(); });
  const W = 640, H = 300, m = { l: 52, r: 16, t: 14, b: 40 }, NMAX = 40;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["band a ± ε", NB.c.blue], ["a_n inside the band", NB.c.blue], ["a_n outside", NB.c.pink], ["smallest N", NB.c.orange, true]]);
  const read = NB.readout(el);

  // smallest N with |a_n - a| < eps for every n >= N (see header for why each bound is enough)
  function smallestN(s, eps) {
    const S = SEQ[s], out = (n) => !(Math.abs(S.f(n) - S.a) < eps);
    if (s === "pm1") return eps > 1 ? 1 : null;
    if (s === "inv" || s === "alt") return Math.floor(1 / eps) + 1;
    if (s === "e") { let n = 1; while (out(n)) n++; return n; }
    const B = Math.ceil(1 / (eps * eps)) + 1;                      // sin: every n > 1/eps^2 is inside
    let last = 0;
    for (let n = 1; n <= B; n++) if (out(n)) last = n;
    return last + 1;
  }

  function draw() {
    NB.clear(svg);
    const S = SEQ[st.seq], eps = st.eps;
    const X = NB.scale(0, NMAX + 1, m.l, W - m.r), Y = NB.scale(S.ylo, S.yhi, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "n", ylabel: "aₙ", xticks: 8, yticks: 5 });
    const clip = (v) => Math.max(S.ylo, Math.min(S.yhi, v));
    const b0 = clip(S.a - eps), b1 = clip(S.a + eps);
    NB.el("rect", { x: X(0), y: Y(b1), width: X(NMAX + 1) - X(0), height: Math.max(0, Y(b0) - Y(b1)), fill: NB.c.blue, "fill-opacity": 0.13 }, svg);
    [S.a - eps, S.a + eps].forEach((v) => { if (v > S.ylo && v < S.yhi) NB.el("line", { x1: X(0), x2: X(NMAX + 1), y1: Y(v), y2: Y(v), stroke: NB.c.blue, "stroke-width": 1, "stroke-opacity": 0.6 }, svg); });
    NB.el("line", { x1: X(0), x2: X(NMAX + 1), y1: Y(S.a), y2: Y(S.a), stroke: NB.c.blue, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    NB.el("text", { x: W - m.r - 2, y: Y(S.a) - 5, "text-anchor": "end", "font-size": 13, fill: NB.c.blue }, svg, "a = " + S.alab);
    const N = smallestN(st.seq, eps);
    if (N !== null && N <= NMAX) {
      NB.el("line", { x1: X(N - 0.5), x2: X(N - 0.5), y1: Y(S.yhi), y2: Y(S.ylo), stroke: NB.c.orange, "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
      NB.el("text", { x: X(N - 0.5) + 5, y: m.t + 14, "font-size": 13, fill: NB.c.orange }, svg, "N = " + N);
    }
    for (let n = 1; n <= NMAX; n++) {
      const v = S.f(n), inside = Math.abs(v - S.a) < eps;
      NB.el("circle", { cx: X(n), cy: Y(clip(v)), r: 4, fill: inside ? NB.c.blue : NB.c.pink, "fill-opacity": N !== null && n >= N ? 1 : 0.85, stroke: "none" }, svg);
    }
    const e = NB.fmt(eps, 3);
    if (S.diverges) {
      read.innerHTML = N === null
        ? `ε = ${e} ≤ 1: <b>no N works</b>, for a = 0 or any other a. The terms 1 and −1 both keep coming back and are 2 apart, so they cannot both lie within ε &lt; 1 of the same a.`
        : `ε = ${e} &gt; 1: every term is within ε of 0, so <b>N = 1</b> works here. Convergence needs an N for <i>every</i> ε, and below ε = 1 there is none.`;
    } else {
      read.innerHTML = `ε = ${e}: smallest <b>N = ${N}</b>${N > NMAX ? " (past the plotted range)" : ""}; every aₙ with n ≥ N is within ε of ${S.alab}. ` +
        `Smaller ε needs a larger N, but some N always exists.`;
    }
  }
  draw();
});

/* fairness-thresholds: post-processing a classifier by choosing one threshold per group. Scores are normal,
 * N(mu1, sd) for people with y = 1 and N(mu0, sd) for y = 0, identical in both groups; only the base rate
 * P(y = 1) differs. Predict y_hat = 1 when score > threshold. The table compares the group-fairness
 * definitions: demographic parity (P(y_hat = 1)), equal opportunity (TPR), equalized odds (TPR and FPR),
 * predictive rate parity (PPV = P(y = 1 | y_hat = 1)). Buttons set group B's threshold to match one of them.
 * params: {baseA: 0.5, baseB: 0.25, tA: 0.5, tB: 0.5} */
NB.register("fairness-thresholds", function (el, P) {
  const mu1 = 0.62, mu0 = 0.38, sd = 0.13;
  const st = { piA: P.baseA ?? 0.5, piB: P.baseB ?? 0.25, tA: P.tA ?? 0.5, tB: P.tB ?? 0.5 };
  const erf = (x) => { // Abramowitz-Stegun 7.1.26
    const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  };
  const above = (t, mu) => 0.5 * (1 - erf((t - mu) / (sd * Math.SQRT2)));
  const stats = (t, pi) => { const tpr = above(t, mu1), fpr = above(t, mu0), pr = pi * tpr + (1 - pi) * fpr; return { pr, tpr, fpr, ppv: pr > 0 ? (pi * tpr) / pr : 0 }; };
  const ctl = NB.controls(el);
  const sA = NB.slider(ctl, { label: "threshold, group A", min: 0, max: 1, step: 0.005, value: st.tA }, (v) => { st.tA = v; draw(); });
  const sB = NB.slider(ctl, { label: "threshold, group B", min: 0, max: 1, step: 0.005, value: st.tB }, (v) => { st.tB = v; draw(); });
  NB.slider(ctl, { label: "base rate P(y=1), group B", min: 0.05, max: 0.95, step: 0.05, value: st.piB }, (v) => { st.piB = v; draw(); });
  const ctl2 = NB.controls(el);
  const match = (metric) => { // bisection on group B's threshold (all metrics are monotone in t over [0, 1])
    const target = stats(st.tA, st.piA)[metric];
    let lo = 0, hi = 1; const dec = stats(0, st.piB)[metric] > stats(1, st.piB)[metric];
    for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2, v = stats(mid, st.piB)[metric]; if ((v > target) === dec) lo = mid; else hi = mid; }
    st.tB = (lo + hi) / 2; sB.set(+st.tB.toFixed(3)); draw();
  };
  NB.button(ctl2, "B matches A's positive rate", () => match("pr"));
  NB.button(ctl2, "B matches A's TPR", () => match("tpr"));
  NB.button(ctl2, "B matches A's PPV", () => match("ppv"));
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W = 340, H = 200, m = { l: 14, r: 10, t: 22, b: 30 };
  const svgs = [NB.svg(row, W, H), NB.svg(row, W, H)];
  NB.legend(el, [["y = 1 (qualified)", NB.c.blue], ["y = 0", NB.c.orange], ["threshold: predict 1 to the right", NB.c.fg, true]]);
  const tbl = NB.el("div", { class: "nb-readout" }, el);
  const pdf = (x, mu) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
  function panel(svg, name, pi, t) {
    NB.clear(svg);
    const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(0, 1.02 * pdf(0, 0), H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "score", grid: false, xticks: 5, yticks: 0, yfmt: () => "" });
    const xs = NB.range(0, 1, 160);
    [[pi, mu1, NB.c.blue], [1 - pi, mu0, NB.c.orange]].forEach(([w, mu, c]) => {
      const pts = xs.map((x) => [x, w * pdf(x, mu)]);
      const right = pts.filter((p) => p[0] >= t);
      if (right.length > 1) NB.el("path", { d: NB.line(right, X, Y) + `L${X(right[right.length - 1][0])},${Y(0)}L${X(right[0][0])},${Y(0)}Z`, fill: c, "fill-opacity": 0.22, stroke: "none" }, svg);
      NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: c, "stroke-width": 2.2 }, svg);
    });
    NB.el("line", { x1: X(t), x2: X(t), y1: Y(0), y2: m.t - 6, stroke: NB.c.fg, "stroke-width": 1.6, "stroke-dasharray": "5 3" }, svg);
    NB.el("text", { x: m.l, y: 14, class: "nb-label" }, svg, `group ${name}: P(y=1) = ${NB.fmt(pi, 2)}`);
  }
  function draw() {
    panel(svgs[0], "A", st.piA, st.tA); panel(svgs[1], "B", st.piB, st.tB);
    const a = stats(st.tA, st.piA), b = stats(st.tB, st.piB);
    const rowf = (lab, k, crit) => { const eq = Math.abs(a[k] - b[k]) < 0.005; return `<tr><td>${lab}</td><td>${crit}</td><td>${NB.fmt(a[k], 2)}</td><td>${NB.fmt(b[k], 2)}</td><td>${eq ? "<b>equal</b>" : "differ by " + NB.fmt(Math.abs(a[k] - b[k]), 2)}</td></tr>`; };
    tbl.innerHTML = `<table style="width:100%;border-collapse:collapse"><tr><th align=left>quantity</th><th align=left>equal under</th><th align=left>A</th><th align=left>B</th><th></th></tr>` +
      rowf("P(ŷ=1)", "pr", "demographic parity") + rowf("TPR = P(ŷ=1 | y=1)", "tpr", "equal opportunity") +
      rowf("FPR = P(ŷ=1 | y=0)", "fpr", "equalized odds (with TPR)") + rowf("PPV = P(y=1 | ŷ=1)", "ppv", "predictive rate parity") + `</table>`;
  }
  draw();
});

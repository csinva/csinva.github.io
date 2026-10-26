/* roc-threshold: a classifier's scores for negatives ~ N(0, 1) and positives ~ N(d, 1), equal numbers of each.
 * Moving the threshold trades true positives against false positives; the ROC curve traces every threshold.
 * params: {sep: 1.5, threshold: 0.5} */
NB.register("roc-threshold", function (el, P) {
  const st = { d: P.sep ?? 1.5, t: P.threshold ?? 0.5 };
  // normal cdf via erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)
  const erf = (x) => { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); };
  const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2)), pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "class separation", min: 0, max: 4, step: 0.05, value: st.d }, (v) => { st.d = v; draw(); });
  NB.slider(ctl, { label: "threshold", min: -3, max: 6, step: 0.05, value: st.t }, (v) => { st.t = v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1.35fr) minmax(0,1fr)", gap: "14px" } }, el);
  const W1 = 470, W2 = 340, H = 320, m = { l: 44, r: 10, t: 10, b: 36 };
  const s1 = NB.svg(row, W1, H), s2 = NB.svg(row, W2, H);
  NB.legend(el, [["negatives (scores)", NB.c.blue], ["positives (scores)", NB.c.orange], ["ROC curve", NB.c.purple]]);
  const read = NB.readout(el);
  const X = NB.scale(-3.5, 7, m.l, W1 - m.r), Y = NB.scale(0, 0.45, H - m.b, m.t);
  const R = NB.scale(0, 1, m.l, W2 - m.r), RY = NB.scale(0, 1, H - m.b, m.t);

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const d = st.d, t = st.t, xs = NB.range(-3.5, 7, 260);
    NB.axes(s1, X, Y, { xlabel: "classifier score", ylabel: "density", xticks: 6, yticks: 3 });
    // shaded tails right of the threshold: FP under the negatives, TP under the positives
    const tail = (mu) => { const pts = xs.filter((x) => x >= t).map((x) => [x, pdf(x - mu)]); if (!pts.length) return "";
      return `M${X(t)},${Y(0)}L${X(t)},${Y(pdf(t - mu))}` + NB.line(pts, X, Y).replace(/^M/, "L") + `L${X(pts[pts.length - 1][0])},${Y(0)}Z`; };
    NB.el("path", { d: tail(d), fill: NB.c.orange, "fill-opacity": 0.22 }, s1);
    NB.el("path", { d: tail(0), fill: NB.c.blue, "fill-opacity": 0.22 }, s1);
    NB.el("path", { d: NB.line(xs.map((x) => [x, pdf(x)]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2 }, s1);
    NB.el("path", { d: NB.line(xs.map((x) => [x, pdf(x - d)]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, s1);
    NB.el("line", { x1: X(t), x2: X(t), y1: Y(0), y2: Y(0.45), stroke: NB.c.fg, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, s1);
    NB.el("text", { x: X(t) + 5, y: Y(0.42), class: "nb-label" }, s1, "predict positive →");
    // ROC
    NB.axes(s2, R, RY, { xlabel: "false positive rate", ylabel: "true positive rate (recall)", xticks: 4, yticks: 4 });
    NB.el("line", { x1: R(0), y1: RY(0), x2: R(1), y2: RY(1), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, s2);
    const roc = NB.range(-6, 10, 300).map((u) => [1 - Phi(u), 1 - Phi(u - d)]);
    NB.el("path", { d: NB.line(roc, R, RY), fill: "none", stroke: NB.c.purple, "stroke-width": 2.2 }, s2);
    const fpr = 1 - Phi(t), tpr = 1 - Phi(t - d);
    NB.el("circle", { cx: R(fpr), cy: RY(tpr), r: 5.5, fill: NB.c.fg }, s2);
    const auc = Phi(d / Math.SQRT2), prec = tpr + fpr > 0 ? tpr / (tpr + fpr) : 1, acc = (tpr + 1 - fpr) / 2, f1 = prec + tpr > 0 ? (2 * prec * tpr) / (prec + tpr) : 0;
    NB.el("text", { x: R(0.55), y: RY(0.12), class: "nb-label" }, s2, `AUC = ${auc.toFixed(3)}`);
    read.innerHTML = `recall (TPR) <b>${tpr.toFixed(2)}</b> · FPR <b>${fpr.toFixed(2)}</b> · specificity <b>${(1 - fpr).toFixed(2)}</b> · precision <b>${prec.toFixed(2)}</b> · accuracy <b>${acc.toFixed(2)}</b> · F-score <b>${f1.toFixed(2)}</b>`;
  }
  draw();
});

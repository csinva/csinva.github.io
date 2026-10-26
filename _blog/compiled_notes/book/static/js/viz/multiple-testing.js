/* multiple-testing: m one-sided z-tests, k of them with a true effect (z ~ N(mu, 1), mu = 3 illustrative), the
 * rest null (z ~ N(0, 1)); p = 1 - Phi(z). The sorted p-values are drawn on a log scale against three cutoffs:
 * no correction (p <= alpha), Bonferroni (p <= alpha / m) and Benjamini-Hochberg (reject the i smallest, where
 * i is the largest rank with p_(i) <= i alpha / m). The readout table repeats the experiment 2000 times and
 * reports the family-wise error rate P(V >= 1), the false discovery rate E[V / max(R, 1)] and the power.
 * Checked against a numpy/scipy simulation (see the report): BH's FDR comes out at alpha * (m - k) / m.
 * params: {m: 50, k: 10, alpha: 0.05, mu: 3, reps: 2000} */
NB.register("multiple-testing", function (el, P) {
  const m = P.m || 50, mu = P.mu ?? 3, reps = P.reps || 2000;
  const st = { k: P.k ?? 10, alpha: P.alpha ?? 0.05, seed: 1 };
  // upper tail of the standard normal, via erfc (Numerical Recipes erfcc: relative error < 1.2e-7 everywhere)
  const erfc = (x) => {
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  };
  const pval = (z) => 0.5 * erfc(z / Math.SQRT2);
  const draw1 = (u, k) => Array.from({ length: m }, (_, i) => { const real = i < k; return { real, p: pval(u.normal() + (real ? mu : 0)) }; })
    .sort((a, b) => a.p - b.p);
  // number of rejections for each procedure, given p-values sorted ascending
  const procs = [
    ["none", "no correction", NB.c.muted, (ps, a) => ps.filter((q) => q.p <= a).length],
    ["bonf", "Bonferroni", NB.c.green, (ps, a) => ps.filter((q) => q.p <= a / m).length],
    ["bh", "Benjamini-Hochberg", NB.c.purple, (ps, a) => { let r = 0; ps.forEach((q, i) => { if (q.p <= ((i + 1) * a) / m) r = i + 1; }); return r; }],
  ];

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "true effects k (of " + m + ")", min: 0, max: m, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; draw(); });
  NB.slider(ctl, { label: "α", min: 0.01, max: 0.2, step: 0.01, value: st.alpha, format: (v) => v.toFixed(2) }, (v) => { st.alpha = v; draw(); });
  NB.button(ctl, "new data", () => { st.seed++; draw(); });
  const W = 640, H = 320, mg = { l: 56, r: 16, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["true effect", NB.c.blue], ["null", NB.c.orange], ["p ≤ α (no correction)", NB.c.muted, true],
                 ["p ≤ α/m (Bonferroni)", NB.c.green, true], ["p ≤ iα/m (BH)", NB.c.purple, true]]);
  const read = NB.readout(el);
  const lo = -6, X = NB.scale(0, m + 1, mg.l, W - mg.r), Y = NB.scale(lo, 0, H - mg.b, mg.t);
  const sup = (k) => String(k).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  const ly = (p) => Math.max(lo, Math.log10(p));

  function draw() {
    NB.clear(svg);
    const a = st.alpha, ps = draw1(NB.rng(1000 + st.seed), st.k);
    NB.axes(svg, X, Y, { xlabel: "rank i of the p-value (smallest first)", ylabel: "p-value (log scale)", xticks: 6, yticks: 6,
      yfmt: (t) => (t === 0 ? "1" : "10" + sup(t)), xfmt: (t) => (t === 0 ? "" : NB.fmt(t)) });
    const nBH = procs[2][3](ps, a);
    if (nBH) NB.el("rect", { x: X(0.5), y: mg.t, width: X(nBH + 0.5) - X(0.5), height: H - mg.t - mg.b, fill: NB.c.purple, "fill-opacity": 0.08 }, svg);
    NB.el("line", { x1: X(0), x2: X(m + 1), y1: Y(Math.log10(a)), y2: Y(Math.log10(a)), stroke: NB.c.muted, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svg);
    NB.el("line", { x1: X(0), x2: X(m + 1), y1: Y(Math.log10(a / m)), y2: Y(Math.log10(a / m)), stroke: NB.c.green, "stroke-dasharray": "5 4", "stroke-width": 2 }, svg);
    NB.el("path", { d: NB.line(NB.range(0.3, m + 1, 200).map((i) => [i, Math.log10((i * a) / m)]), X, Y), fill: "none", stroke: NB.c.purple, "stroke-dasharray": "5 4", "stroke-width": 2 }, svg);
    ps.forEach((q, i) => {
      const c = q.real ? NB.c.blue : NB.c.orange, rej = i < nBH;
      NB.el("circle", { cx: X(i + 1), cy: Y(ly(q.p)), r: 4.2, fill: rej ? c : "none", stroke: c, "stroke-width": 1.6 }, svg);
    });
    if (nBH) NB.el("text", { x: X(0.5) + 4, y: mg.t + 14, style: { fontSize: "12px", fill: NB.c.purple } }, svg, `BH rejects these ${nBH}`);

    // repeated experiments: FWER, FDR and power for each procedure
    const u = NB.rng(77 + st.k * 1009 + Math.round(a * 100)), acc = procs.map(() => ({ fw: 0, fdp: 0, pow: 0 }));
    for (let r = 0; r < reps; r++) {
      const q = draw1(u, st.k);
      procs.forEach((pr, j) => {
        const R = pr[3](q, a); let V = 0; for (let i = 0; i < R; i++) if (!q[i].real) V++;
        acc[j].fw += V > 0; acc[j].fdp += R ? V / R : 0; acc[j].pow += st.k ? (R - V) / st.k : 0;
      });
    }
    const pc = (v) => (100 * v / reps).toFixed(1) + "%";
    const rows = procs.map((pr, j) => {
      const R = pr[3](ps, a); let V = 0; for (let i = 0; i < R; i++) if (!ps[i].real) V++;
      return `<tr><td style="color:${pr[2]};font-weight:600;padding-right:10px">${pr[1]}</td><td><b>${R}</b> (${V} false)</td>` +
        `<td><b>${pc(acc[j].fw)}</b></td><td><b>${pc(acc[j].fdp)}</b></td><td>${st.k ? pc(acc[j].pow) : "–"}</td></tr>`;
    }).join("");
    read.innerHTML = `<table style="border-collapse:collapse;margin:2px 0;font-size:inherit" cellpadding="2">` +
      `<tr><td></td><td>discoveries here</td><td>FWER</td><td>FDR</td><td>power</td></tr>${rows}</table>` +
      `FWER, FDR and power are averages over ${reps} repeats of the experiment. With independent tests BH keeps the FDR at α(m − k)/m = <b>${NB.fmt((a * (m - st.k)) / m, 3)}</b>.`;
  }
  draw();
});

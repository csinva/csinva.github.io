/* missing-data: the three missingness mechanisms (Little & Rubin) and what they do to an estimate of E[Y].
 * n = 300 rows with X ~ N(0, 1) and Y = 2 + X + N(0, 0.7^2), so E[Y] = 2. X is always observed; Y is missing with
 * probability  sigmoid(a + s z),  where z = 0 (MCAR), z = X (MAR) or z = standardized Y (MNAR), s is the slider and
 * a is set so that about 40% of Y is missing on average. Each row keeps one uniform draw u_i and is missing when
 * u_i < P(missing), so moving the slider changes the missing set smoothly.
 * Two estimates of E[Y]: drop the incomplete rows (complete-case mean), and regression imputation (fit Y ~ X on the
 * complete rows, fill each missing Y with its prediction, average all n). Regression imputation is unbiased under
 * MAR because, given X, the complete rows have the same distribution of Y as the missing ones; under MNAR they don't.
 * params: {mech: "mar", s: 2} */
NB.register("missing-data", function (el, P) {
  const n = 300, rng = NB.rng(41);
  for (let k = 0; k < 5; k++) rng();
  const rows = Array.from({ length: n }, () => {
    const x = rng.normal(), y = 2 + x + 0.7 * rng.normal();
    return { x, y, u: rng() };
  });
  const sdY = Math.sqrt(1 + 0.49);
  const st = { mech: P.mech || "mar", s: P.s ?? 2 };
  const ctl = NB.controls(el);
  NB.select(ctl, "mechanism", [["mcar", "MCAR: unrelated to any value"], ["mar", "MAR: depends on observed X"], ["mnar", "MNAR: depends on Y itself"]], st.mech, (v) => { st.mech = v; draw(); });
  NB.slider(ctl, { label: "how strongly missingness depends on it", min: 0, max: 4, step: 0.1, value: st.s }, (v) => { st.s = v; draw(); });
  const narrow = el.clientWidth > 0 && el.clientWidth < 520;   // phones: a narrower coordinate system, so text stays readable
  const W = narrow ? 420 : 640, H = narrow ? 340 : 330, m = { l: 40, r: narrow ? 104 : 150, t: 12, b: 38 }, ext = narrow ? 12 : 40;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["observed (X, Y)", NB.c.blue], ["Y missing (true value, never seen)", NB.c.muted, true], ["imputed Y", NB.c.green]]);
  const read = NB.readout(el);
  const X = NB.scale(-3, 3, m.l, W - m.r), Y = NB.scale(-2.5, 6.5, H - m.b, m.t);
  const sig = (t) => 1 / (1 + Math.exp(-t));
  // offset a so that the average missing probability is 0.4 under each setting (bisection on the population average,
  // approximated by the average over these 300 rows)
  function offset(zs, s) {
    let lo = -10, hi = 10;
    for (let it = 0; it < 60; it++) {
      const a = (lo + hi) / 2, mean = zs.reduce((t, z) => t + sig(a + s * z), 0) / zs.length;
      if (mean > 0.4) hi = a; else lo = a;
    }
    return (lo + hi) / 2;
  }
  function draw() {
    NB.clear(svg);
    const s = st.mech === "mcar" ? 0 : st.s;
    const zs = rows.map((r) => (st.mech === "mar" ? r.x : st.mech === "mnar" ? (r.y - 2) / sdY : 0));
    const a = offset(zs, s);
    const miss = rows.map((r, i) => r.u < sig(a + s * zs[i]));
    const obs = rows.filter((_, i) => !miss[i]);
    const mean = (v) => v.reduce((t, x) => t + x, 0) / v.length;
    const full = mean(rows.map((r) => r.y)), cc = mean(obs.map((r) => r.y));
    const xb = mean(obs.map((r) => r.x)), yb = cc;
    const b1 = obs.reduce((t, r) => t + (r.x - xb) * (r.y - yb), 0) / obs.reduce((t, r) => t + (r.x - xb) ** 2, 0), b0 = yb - b1 * xb;
    const imp = mean(rows.map((r, i) => (miss[i] ? b0 + b1 * r.x : r.y)));

    NB.axes(svg, X, Y, { xlabel: "X (always observed)", ylabel: "Y", xticks: 6, yticks: 5 });
    rows.forEach((r, i) => {
      if (miss[i]) NB.el("circle", { cx: X(r.x), cy: Y(r.y), r: 2.6, fill: "none", stroke: NB.c.muted, "stroke-width": 0.9 }, svg);
      else NB.el("circle", { cx: X(r.x), cy: Y(r.y), r: 2.6, fill: NB.c.blue, "fill-opacity": 0.55 }, svg);
    });
    rows.forEach((r, i) => { if (miss[i]) NB.el("circle", { cx: X(r.x), cy: Y(b0 + b1 * r.x), r: 1.8, fill: NB.c.green }, svg); });
    NB.el("path", { d: NB.line([[-3, b0 - 3 * b1], [3, b0 + 3 * b1]], X, Y), stroke: NB.c.green, "stroke-width": 1.2, fill: "none", "stroke-opacity": 0.7 }, svg);
    // the three estimates of E[Y] as horizontal ticks in the right margin
    const x0 = W - m.r + 8, x1 = W - m.r + ext;
    const marks = [[full, NB.c.fg, "all rows", true], [cc, NB.c.orange, "drop rows"], [imp, NB.c.green, "impute"]];
    marks.sort((p, q) => q[0] - p[0]);
    let lastY = -1e9;
    marks.forEach(([v, c, lab, dash]) => {
      NB.el("line", { x1: m.l, x2: x1, y1: Y(v), y2: Y(v), stroke: c, "stroke-width": dash ? 1.4 : 2.2, "stroke-dasharray": dash ? "5 4" : null, "stroke-opacity": dash ? 0.8 : 1 }, svg);
      const ty = Math.max(Y(v) + 4, lastY + 14); lastY = ty;
      NB.el("text", { x: x1 + 4, y: ty, class: "nb-label", style: { fill: c, fontWeight: 600 } }, svg, `${lab} ${NB.fmt(v, 2)}`);
    });
    NB.el("text", { x: x0 - 4, y: m.t + 10, class: "nb-tick" }, svg, "estimates of E[Y]");
    const sgn = (v) => (v >= 0 ? "+" : "−") + NB.fmt(Math.abs(v), 2);
    read.innerHTML = `${n - obs.length} of ${n} values of Y missing. Mean of Y over all rows (unknowable in practice): <b>${NB.fmt(full, 2)}</b>. ` +
      `Dropping incomplete rows: <b>${NB.fmt(cc, 2)}</b> (${sgn(cc - full)}). Regression imputation: <b>${NB.fmt(imp, 2)}</b> (${sgn(imp - full)}).`;
  }
  draw();
});

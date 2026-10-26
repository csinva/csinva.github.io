/* shapley-values: exact Shapley values for a 3-feature model with one interaction,
 *   f(x) = 1 + 2 x1 + x2 - 1.5 x3 + γ x1 x2,
 * explained against the baseline b = (0, 0, 0). A feature "missing" from a coalition S takes its baseline
 * value, so v(S) = f(x_S, b_rest). φ_j averages the change f gains when j is added, over all 3! = 6 orders.
 * Top: waterfall from f(b) to f(x). Below: the marginal contribution of each feature in each order.
 * Checked against a brute-force python enumeration and the closed form φ1 = 2x1 + γx1x2/2, etc.
 * params: {x1: 1, x2: 1.5, x3: 0.5, gamma: 2} */
NB.register("shapley-values", function (el, P) {
  const st = { x: [P.x1 ?? 1, P.x2 ?? 1.5, P.x3 ?? 0.5], g: P.gamma ?? 2 };
  const base = [0, 0, 0];
  const f = (z) => 1 + 2 * z[0] + z[1] - 1.5 * z[2] + st.g * z[0] * z[1];
  const v = (S) => f(base.map((b, j) => (S.includes(j) ? st.x[j] : b)));     // coalition value
  const orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const cols = [NB.c.blue, NB.c.orange, NB.c.green];
  const sub = ["₁", "₂", "₃"];

  const ctl = NB.controls(el);
  [0, 1, 2].forEach((j) => NB.slider(ctl, { label: "x" + sub[j], min: -2, max: 2, step: 0.1, value: st.x[j], format: (t) => t.toFixed(1) }, (t) => { st.x[j] = t; draw(); }));
  NB.slider(ctl, { label: "interaction γ", min: -3, max: 3, step: 0.1, value: st.g, format: (t) => t.toFixed(1) }, (t) => { st.g = t; draw(); });
  const eq = NB.el("div", { class: "nb-readout", style: { marginBottom: "4px" } }, el);
  const W = 640, H = 270, m = { l: 48, r: 14, t: 20, b: 40 };
  const svg = NB.svg(el, W, H);
  const tbl = NB.el("table", { class: "nb-shap-table", style: { width: "100%", fontSize: "0.86rem", fontVariantNumeric: "tabular-nums", margin: "8px 0", borderCollapse: "collapse" } }, el);
  const read = NB.readout(el);
  const sgn = (t) => (t >= 0 ? "+" : "−") + NB.fmt(Math.abs(t), 2);

  function draw() {
    const x = st.x;
    // marginal contribution of each feature in each order
    const rows = orders.map((o) => { const d = [0, 0, 0], S = []; o.forEach((j) => { const before = v(S); S.push(j); d[j] = v(S) - before; }); return { o, d }; });
    const phi = [0, 1, 2].map((j) => rows.reduce((s, r) => s + r.d[j], 0) / rows.length);
    const fb = v([]), fx = v([0, 1, 2]);
    eq.innerHTML = `f(x) = 1 + 2x₁ + x₂ − 1.5x₃ + γ·x₁x₂, baseline b = (0, 0, 0). ` +
      `At x = (${x.map((t) => NB.fmt(t, 1)).join(", ")}): f(b) = <b>${NB.fmt(fb, 2)}</b>, f(x) = <b>${NB.fmt(fx, 2)}</b>.`;

    // waterfall
    NB.clear(svg);
    const steps = [{ lab: "f(b)", lo: 0, hi: fb, c: NB.c.muted, val: fb }];
    let run = fb;
    [0, 1, 2].forEach((j) => { steps.push({ lab: "φ" + sub[j] + " (x" + sub[j] + ")", lo: run, hi: run + phi[j], c: cols[j], val: phi[j], delta: true }); run += phi[j]; });
    steps.push({ lab: "f(x)", lo: 0, hi: fx, c: NB.c.fg, val: fx });
    const all = steps.flatMap((s) => [s.lo, s.hi]);
    let y0 = Math.min(0, ...all), y1 = Math.max(0, ...all); const pad = 0.12 * (y1 - y0 || 1); y0 -= y0 < 0 ? pad : 0; y1 += pad;
    const X = NB.scale(0, steps.length, m.l, W - m.r), Y = NB.scale(y0, y1, H - m.b, m.t);
    NB.axes(svg, X, Y, { yticks: 5, xticks: 1 });
    svg.querySelectorAll(".nb-axes text").forEach((t) => { if (+t.getAttribute("y") > H - m.b + 4) t.remove(); });    // drop numeric x ticks
    svg.querySelectorAll(".nb-axes line.nb-grid").forEach((l) => { if (l.getAttribute("x1") === l.getAttribute("x2")) l.remove(); });
    NB.el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: NB.c.muted }, svg);
    const bw = (X(1) - X(0)) * 0.56;
    steps.forEach((s, i) => {
      const cx = X(i + 0.5), top = Y(Math.max(s.lo, s.hi)), bot = Y(Math.min(s.lo, s.hi));
      NB.el("rect", { x: cx - bw / 2, y: top, width: bw, height: Math.max(1, bot - top), fill: s.c, "fill-opacity": s.delta ? 0.75 : 0.35, stroke: s.c }, svg);
      if (i < steps.length - 1) NB.el("line", { x1: cx + bw / 2, x2: X(i + 1.5) - bw / 2, y1: Y(s.hi), y2: Y(s.hi), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, svg);
      const up = s.hi >= s.lo;
      NB.el("text", { x: cx, y: up ? top - 5 : bot + 14, class: "nb-tick", "text-anchor": "middle" }, svg, s.delta ? sgn(s.val) : NB.fmt(s.val, 2));
      NB.el("text", { x: cx, y: H - m.b + 18, class: "nb-label", "text-anchor": "middle" }, svg, s.lab);
    });

    // table of marginal contributions
    NB.clear(tbl);
    const hr = NB.el("tr", {}, NB.el("thead", {}, tbl));
    NB.el("th", {}, hr, "order features are added");
    [0, 1, 2].forEach((j) => NB.el("th", { style: { color: cols[j], textAlign: "right" } }, hr, "adding x" + sub[j]));
    const tb = NB.el("tbody", {}, tbl);
    rows.forEach((r) => {
      const tr = NB.el("tr", {}, tb);
      NB.el("td", {}, tr, r.o.map((j) => "x" + sub[j]).join(" → "));
      r.d.forEach((d) => NB.el("td", { style: { textAlign: "right" } }, tr, sgn(d)));
    });
    const fr = NB.el("tr", { style: { fontWeight: 600 } }, NB.el("tfoot", {}, tbl));
    NB.el("td", {}, fr, "average = Shapley value φⱼ");
    phi.forEach((p, j) => NB.el("td", { style: { textAlign: "right", color: cols[j] } }, fr, sgn(p)));

    const sum = phi[0] + phi[1] + phi[2], inter = st.g * x[0] * x[1];
    read.innerHTML = `φ₁ + φ₂ + φ₃ = <b>${NB.fmt(sum, 2)}</b> = f(x) − f(b) = ${NB.fmt(fx, 2)} − ${NB.fmt(fb, 2)} (local accuracy). ` +
      `The interaction term γx₁x₂ = ${NB.fmt(inter, 2)} shows up only when x₁ and x₂ are both present, so it goes to whichever of them is added second; ` +
      `averaged over the 6 orders it is split evenly, ${NB.fmt(inter / 2, 2)} each.` +
      (Math.abs(x[2]) < 1e-9 ? " x₃ equals its baseline, so φ₃ = 0." : "");
  }
  draw();
});

/* uniform-convergence: a sequence of functions f_n and its pointwise limit f, with an ε-band around f.
 * Uniform convergence means that for large n the whole graph of f_n fits inside the band.
 * Parts of f_n outside the band are drawn in pink.
 * params: {seq: "pow" | "powc" | "bump", n: 5, eps: 0.1} */
NB.register("uniform-convergence", function (el, P) {
  const seqs = {
    pow: { label: "xⁿ on [0, 1]", f: (x, n) => Math.pow(x, n), lim: (x) => (x < 1 ? 0 : 1), a: 0, b: 1,
           sup: () => 1, note: "the limit jumps to 1 at x = 1, and fₙ gets close to 1 just left of 1 for every n" },
    powc: { label: "xⁿ on [0, 0.9]", f: (x, n) => Math.pow(x, n), lim: () => 0, a: 0, b: 0.9,
            sup: (n) => Math.pow(0.9, n), note: "the sup is 0.9ⁿ, at the right end" },
    bump: { label: "nx / (1 + n²x²) on [0, 1]", f: (x, n) => (n * x) / (1 + n * n * x * x), lim: () => 0, a: 0, b: 1,
            sup: () => 0.5, note: "the bump keeps height ½ and moves to x = 1/n" },
  };
  const st = { seq: P.seq || "pow", n: P.n ?? 5, eps: P.eps ?? 0.1 };
  const ctl = NB.controls(el);
  NB.select(ctl, "sequence", Object.entries(seqs).map(([k, v]) => [k, v.label]), st.seq, (v) => { st.seq = v; draw(); });
  NB.slider(ctl, { label: "n", min: 1, max: 100, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  NB.slider(ctl, { label: "ε", min: 0.02, max: 0.6, step: 0.01, value: st.eps }, (v) => { st.eps = v; draw(); });
  const W = 640, H = 300, m = { l: 44, r: 12, t: 10, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["limit f", NB.c.orange], ["ε-band around f", NB.c.orange, true], ["fₙ inside the band", NB.c.blue], ["fₙ outside", NB.c.pink]]);
  const read = NB.readout(el);

  function draw() {
    NB.clear(svg);
    const S = seqs[st.seq], X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(-0.65, 1.2, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "x", xticks: 5, yticks: 5 });
    const xs = NB.range(S.a, S.b, 801);
    const endOpen = st.seq === "pow";              // the limit of xⁿ on [0, 1] is 0 on [0, 1) and 1 at x = 1
    const band = endOpen ? xs.slice(0, -1) : xs;
    const top = band.map((x) => [x, S.lim(x) + st.eps]), bot = band.map((x) => [x, S.lim(x) - st.eps]).reverse();
    NB.el("path", { d: NB.line(top.concat(bot), X, Y) + "Z", fill: NB.c.orange, "fill-opacity": 0.12, stroke: NB.c.orange, "stroke-dasharray": "4 4", "stroke-opacity": 0.7 }, svg);
    NB.el("path", { d: NB.line(band.map((x) => [x, S.lim(x)]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg);
    if (endOpen) {
      NB.el("circle", { cx: X(1), cy: Y(0), r: 4, fill: NB.c.bg, stroke: NB.c.orange, "stroke-width": 2 }, svg);
      NB.el("circle", { cx: X(1), cy: Y(1), r: 4, fill: NB.c.orange }, svg);
    }
    // faint earlier terms, then f_n split into inside / outside the band
    [1, 2, 4, 8, 16, 32, 64].filter((k) => k < st.n).forEach((k) =>
      NB.el("path", { d: NB.line(xs.map((x) => [x, S.f(x, k)]), X, Y), fill: "none", stroke: NB.c.muted, "stroke-opacity": 0.3 }, svg));
    let segs = [], cur = null;
    xs.forEach((x) => {
      const y = S.f(x, st.n), out = Math.abs(y - S.lim(x)) >= st.eps;
      if (!cur || cur.out !== out) { if (cur) cur.pts.push([x, y]); cur = { out, pts: cur ? [cur.pts[cur.pts.length - 1]] : [] }; segs.push(cur); }
      cur.pts.push([x, y]);
    });
    segs.forEach((s) => NB.el("path", { d: NB.line(s.pts, X, Y), fill: "none", stroke: s.out ? NB.c.pink : NB.c.blue, "stroke-width": 2.6 }, svg));
    if (S.b < 1) NB.el("line", { x1: X(S.b), x2: X(S.b), y1: Y(-0.65), y2: Y(1.2), stroke: NB.c.muted, "stroke-dasharray": "2 4" }, svg);
    const sup = S.sup(st.n), inside = sup < st.eps;
    let nNeeded = "";
    if (st.seq === "powc") nNeeded = ` · inside for every n ≥ <b>${Math.max(1, Math.ceil(Math.log(st.eps) / Math.log(0.9) + 1e-9))}</b>`;
    else nNeeded = ` · no n works once ε ≤ ${st.seq === "pow" ? "1" : "½"}: not uniform`;
    read.innerHTML = `sup |fₙ − f| = <b>${NB.fmt(sup, 3)}</b> (${S.note}) · f<sub>${st.n}</sub> ${inside ? "<b>fits inside</b>" : "<b>leaves</b>"} the ε-band${nNeeded}`;
  }
  draw();
});

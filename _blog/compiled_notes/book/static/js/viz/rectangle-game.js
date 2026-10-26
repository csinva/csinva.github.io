/* rectangle-game: Kearns' rectangle learning game with D uniform on the unit square. The learner outputs the
 * tightest rectangle around the positive examples; its error is the target area it misses. Right: over many
 * repeated games, how often the error exceeds ε, next to the bound 4(1 - ε/4)^m.
 * params: {m: 30, eps: 0.1} */
NB.register("rectangle-game", function (el, P) {
  const R = [0.2, 0.25, 0.75, 0.7];   // target concept c: [x0, y0, x1, y1]
  const area = (r) => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
  const st = { m: P.m || 30, eps: P.eps || 0.1, seed: 1 };
  const inside = (p) => p[0] >= R[0] && p[0] <= R[2] && p[1] >= R[1] && p[1] <= R[3];
  const fit = (pts) => { const pos = pts.filter(inside); if (!pos.length) return null;
    return [Math.min(...pos.map((p) => p[0])), Math.min(...pos.map((p) => p[1])), Math.max(...pos.map((p) => p[0])), Math.max(...pos.map((p) => p[1]))]; };
  const err = (h) => area(R) - (h ? area(h) : 0);   // h sits inside R, so it errs exactly on R \ h
  // error of the tightest fit over many repeated games, for a grid of sample sizes (computed once)
  const ms = Array.from({ length: 41 }, (_, i) => Math.max(1, i * 5)), trials = 400, rng = NB.rng(4);
  const errs = ms.map((m) => Array.from({ length: trials }, () => err(fit(Array.from({ length: m }, () => [rng(), rng()])))));

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "examples m", min: 1, max: 200, step: 1, value: st.m, format: (v) => v.toFixed(0) }, (v) => { st.m = v; draw(); });
  NB.slider(ctl, { label: "ε", min: 0.02, max: 0.3, step: 0.01, value: st.eps }, (v) => { st.eps = v; draw(); });
  NB.button(ctl, "draw a new sample", () => { st.seed++; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.3fr)", gap: "14px" } }, el);
  const S = 320, s1 = NB.svg(row, S, S), W2 = 420, s2 = NB.svg(row, W2, S);
  NB.legend(el, [["target c", NB.c.fg, true], ["learned h (tightest fit)", NB.c.orange], ["P(error > ε), simulated", NB.c.blue], ["bound 4(1 − ε/4)^m", NB.c.pink]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, 10, S - 10), Y = NB.scale(0, 1, S - 10, 10);
  const M = NB.scale(0, 200, 44, W2 - 10), Q = NB.scale(0, 1, S - 36, 10);

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const r = NB.rng(100 + st.seed), pts = Array.from({ length: st.m }, () => [r(), r()]), h = fit(pts), e = err(h);
    NB.el("rect", { x: X(0), y: Y(1), width: X(1) - X(0), height: Y(0) - Y(1), fill: "none", stroke: NB.c.grid }, s1);
    NB.el("rect", { x: X(R[0]), y: Y(R[3]), width: X(R[2]) - X(R[0]), height: Y(R[1]) - Y(R[3]), fill: NB.c.pink, "fill-opacity": 0.28, stroke: NB.c.fg, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, s1);
    if (h) NB.el("rect", { x: X(h[0]), y: Y(h[3]), width: X(h[2]) - X(h[0]), height: Y(h[1]) - Y(h[3]), fill: NB.c.bg, stroke: NB.c.orange, "stroke-width": 2.2 }, s1);
    if (h) NB.el("rect", { x: X(h[0]), y: Y(h[3]), width: X(h[2]) - X(h[0]), height: Y(h[1]) - Y(h[3]), fill: NB.c.orange, "fill-opacity": 0.15 }, s1);
    pts.forEach((p) => inside(p)
      ? NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3.4, fill: NB.c.green }, s1)
      : NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3, fill: "none", stroke: NB.c.muted, "stroke-width": 1.2 }, s1));
    // failure probability vs m
    NB.axes(s2, M, Q, { xlabel: "examples m", ylabel: "probability", xticks: 4, yticks: 4 });
    const fail = errs.map((E) => E.filter((v) => v > st.eps).length / trials);
    NB.el("path", { d: NB.line(ms.map((m) => [m, Math.min(1, 4 * Math.pow(1 - st.eps / 4, m))]), M, Q), fill: "none", stroke: NB.c.pink, "stroke-width": 2 }, s2);
    NB.el("path", { d: NB.line(ms.map((m, i) => [m, fail[i]]), M, Q), fill: "none", stroke: NB.c.blue, "stroke-width": 2.2 }, s2);
    NB.el("line", { x1: M(st.m), x2: M(st.m), y1: Q(0), y2: Q(1), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, s2);
    const need = Math.ceil(Math.log(0.05 / 4) / Math.log(1 - st.eps / 4));
    read.innerHTML = `this sample: ${pts.filter(inside).length} positives, error of h = <b>${NB.fmt(e, 3)}</b> (${e > st.eps ? "more" : "less"} than ε) · for δ = 0.05 the bound asks for m ≥ <b>${need}</b>`;
  }
  draw();
});

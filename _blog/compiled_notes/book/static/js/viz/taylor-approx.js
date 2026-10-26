/* taylor-approx: Taylor polynomials of a function around a point x0.
 * The 0th-, 1st- and 2nd-order approximations use the chapter's colors (blue, orange, green);
 * the order-N polynomial from the slider is drawn in purple.
 * params: {fn: "sin" | "exp" | "log", x0: 0, order: 3} */
NB.register("taylor-approx", function (el, P) {
  const fact = (k) => { let r = 1; for (let i = 2; i <= k; i++) r *= i; return r; };
  const fns = {
    sin: { label: "sin(x)", f: Math.sin, d: (k, a) => [Math.sin, Math.cos, (x) => -Math.sin(x), (x) => -Math.cos(x)][k % 4](a),
           box: [-7, 7, -2.2, 2.2], x0: [-6, 6] },
    exp: { label: "eˣ", f: Math.exp, d: (k, a) => Math.exp(a), box: [-4, 3, -1.5, 12], x0: [-3, 2] },
    log: { label: "ln(1 + x)", f: (x) => (x > -1 ? Math.log(1 + x) : NaN),
           d: (k, a) => (k === 0 ? Math.log(1 + a) : ((k % 2 ? 1 : -1) * fact(k - 1)) / Math.pow(1 + a, k)),
           box: [-0.99, 3, -3, 2.2], x0: [-0.5, 1.5] },
  };
  const st = { fn: P.fn || "sin", x0: P.x0 ?? 0, n: P.order ?? 3 };
  const ctl = NB.controls(el);
  NB.select(ctl, "function", Object.entries(fns).map(([k, v]) => [k, v.label]), st.fn, (v) => {
    st.fn = v; const r = fns[v].x0; st.x0 = Math.max(r[0], Math.min(r[1], st.x0));
    x0s.input.min = r[0]; x0s.input.max = r[1]; x0s.set(st.x0); draw();
  });
  const x0s = NB.slider(ctl, { label: "expansion point x₀", min: fns[st.fn].x0[0], max: fns[st.fn].x0[1], step: 0.05, value: st.x0 }, (v) => { st.x0 = v; draw(); });
  NB.slider(ctl, { label: "order N", min: 0, max: 15, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  const W = 640, H = 320, m = { l: 44, r: 10, t: 10, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["f(x)", NB.c.muted], ["0th order", NB.c.blue, true], ["1st order", NB.c.orange, true], ["2nd order", NB.c.green, true], ["order N", NB.c.purple]]);
  const read = NB.readout(el);
  const clipId = "tc" + Math.random().toString(36).slice(2, 8);

  function poly(F, a, N) { const c = []; for (let k = 0; k <= N; k++) c.push(F.d(k, a) / fact(k)); return (x) => c.reduce((s, ck, k) => s + ck * Math.pow(x - a, k), 0); }
  function draw() {
    NB.clear(svg);
    const F = fns[st.fn], [x0, x1, y0, y1] = F.box;
    const X = NB.scale(x0, x1, m.l, W - m.r), Y = NB.scale(y0, y1, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "x", xticks: 7, yticks: 5 });
    NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, NB.el("clipPath", { id: clipId }, NB.el("defs", {}, svg)));
    const g = NB.el("g", { "clip-path": `url(#${clipId})` }, svg);
    const xs = NB.range(x0, x1, 500), pad = y1 - y0;
    const curve = (fn) => NB.line(xs.map((x) => [x, Math.max(y0 - pad, Math.min(y1 + pad, fn(x)))]), X, Y);
    NB.el("path", { d: curve(F.f), fill: "none", stroke: NB.c.muted, "stroke-width": 3.5, "stroke-opacity": 0.55 }, g);
    [[0, NB.c.blue], [1, NB.c.orange], [2, NB.c.green]].forEach(([k, c]) =>
      NB.el("path", { d: curve(poly(F, st.x0, k)), fill: "none", stroke: c, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, g));
    const TN = poly(F, st.x0, st.n);
    NB.el("path", { d: curve(TN), fill: "none", stroke: NB.c.purple, "stroke-width": 2.4 }, g);
    NB.el("circle", { cx: X(st.x0), cy: Y(F.f(st.x0)), r: 4.5, fill: NB.c.fg }, g);
    // the interval around x0 where the order-N polynomial stays within 0.05 of f
    const reach = (dir) => { let x = st.x0; while (x + dir * 0.005 > x0 && x + dir * 0.005 < x1 && Math.abs(TN(x + dir * 0.005) - F.f(x + dir * 0.005)) < 0.05) x += dir * 0.005; return x; };
    const lo = reach(-1), hi = reach(1);
    NB.el("line", { x1: X(lo), x2: X(hi), y1: H - m.b - 4, y2: H - m.b - 4, stroke: NB.c.purple, "stroke-width": 5, "stroke-linecap": "round", "stroke-opacity": 0.6 }, svg);
    read.innerHTML = `order ${st.n} around x₀ = ${NB.fmt(st.x0, 2)} stays within 0.05 of f on <b>[${NB.fmt(lo, 2)}, ${NB.fmt(hi, 2)}]</b> (purple bar on the x-axis)`;
  }
  draw();
});

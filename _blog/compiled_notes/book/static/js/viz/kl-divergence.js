/* kl-divergence: for a binary variable with true distribution p = P(X = 1), coding with q costs the cross-entropy
 * H_q(p) = E_p[-log q(X)] bits on average. Plotted against q, it touches the entropy H(p) only at q = p; the gap is
 * D(p||q) = H_q(p) - H(p). Colors follow the chapter: E_p[-log q] blue, H(p) orange.
 * params: {p: 0.3, q: 0.6} */
NB.register("kl-divergence", function (el, P) {
  const lg = (x) => (x > 0 ? Math.log2(x) : 0);
  const H = (p) => -(p * lg(p) + (1 - p) * lg(1 - p));
  const CE = (p, q) => -(p * Math.log2(q) + (1 - p) * Math.log2(1 - q));
  const st = { p: P.p ?? 0.3, q: P.q ?? 0.6 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "true p = P(X = 1)", min: 0.01, max: 0.99, step: 0.01, value: st.p }, (v) => { st.p = v; draw(); });
  NB.slider(ctl, { label: "code's q", min: 0.01, max: 0.99, step: 0.01, value: st.q }, (v) => { st.q = v; draw(); });
  const W = 640, H0 = 290, m = { l: 44, r: 10, t: 10, b: 34 }, ymax = 3;
  const svg = NB.svg(el, W, H0);
  NB.legend(el, [["cross-entropy H_q(p) = E_p[-log q(X)], as q varies", NB.c.blue], ["entropy H(p)", NB.c.orange], ["D(p||q), the extra bits", NB.c.green]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(0, ymax, H0 - m.b, m.t);
  function draw() {
    NB.clear(svg);
    NB.axes(svg, X, Y, { xlabel: "q, the distribution the code is built for", ylabel: "bits per symbol", xticks: 5, yticks: 3 });
    const qs = NB.range(0.005, 0.995, 300);
    NB.el("path", { d: NB.line(qs.map((q) => [q, Math.min(CE(st.p, q), ymax)]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.2 }, svg);
    NB.el("line", { x1: X(0), x2: X(1), y1: Y(H(st.p)), y2: Y(H(st.p)), stroke: NB.c.orange, "stroke-width": 2 }, svg);
    NB.el("line", { x1: X(st.p), x2: X(st.p), y1: Y(0), y2: Y(H(st.p)), stroke: NB.c.orange, "stroke-dasharray": "3 3" }, svg);
    NB.el("text", { x: X(st.p), y: Y(0) - 4, class: "nb-tick", "text-anchor": "middle" }, svg, "q = p");
    const ce = CE(st.p, st.q), h = H(st.p);
    NB.el("line", { x1: X(st.q), x2: X(st.q), y1: Y(h), y2: Y(Math.min(ce, ymax)), stroke: NB.c.green, "stroke-width": 4 }, svg);
    NB.el("circle", { cx: X(st.q), cy: Y(Math.min(ce, ymax)), r: 4.5, fill: NB.c.blue }, svg);
    NB.el("text", { x: X(st.q) + (st.q > 0.85 ? -8 : 8), y: (Y(h) + Y(Math.min(ce, ymax))) / 2 + 4, class: "nb-label", "text-anchor": st.q > 0.85 ? "end" : "start", style: { fill: "var(--nb-c-green)" } }, svg, "D(p||q)");
    const dpq = ce - h, dqp = CE(st.q, st.p) - H(st.q);
    read.innerHTML = `H(p) = <b>${NB.fmt(h, 3)}</b> bits · H_q(p) = <b>${NB.fmt(ce, 3)}</b> · D(p||q) = <b>${NB.fmt(dpq, 3)}</b> · swapped, D(q||p) = <b>${NB.fmt(dqp, 3)}</b>`;
  }
  draw();
});

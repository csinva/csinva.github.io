/* center-surround: an on-center receptive field as a difference of Gaussians (narrow positive center,
 * wide negative surround, each with unit volume). The response to a centered light spot of radius r is
 *   R(r) = (1 − exp(−r²/2σc²)) − k (1 − exp(−r²/2σs²)),
 * so it grows while the spot fills the center, then falls as the spot spreads into the surround.
 * params: {radius: 1, k: 0.8, sc: 1, ss: 3} */
NB.register("center-surround", function (el, P) {
  const st = { r: P.radius || 1, k: P.k === undefined ? 0.8 : P.k };
  const sc = P.sc || 1, ss = P.ss || 3, RMAX = 8;
  const R = (r, k) => (1 - Math.exp(-(r * r) / (2 * sc * sc))) - k * (1 - Math.exp(-(r * r) / (2 * ss * ss)));
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "spot radius", min: 0, max: RMAX, step: 0.1, value: st.r, format: (v) => v.toFixed(1) }, (v) => { st.r = v; draw(); });
  NB.slider(ctl, { label: "surround strength k", min: 0, max: 1, step: 0.05, value: st.k }, (v) => { st.k = v; draw(); });
  const row = NB.el("div", { class: "nb-row", style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.5fr)", gap: "14px" } }, el);
  const s1 = NB.svg(row, 300, 300), s2 = NB.svg(row, 420, 300);
  NB.legend(el, [["center (+)", NB.c.blue], ["surround (−)", NB.c.orange], ["light spot", NB.c.yellow]]);
  const read = NB.readout(el);
  const m = { l: 46, r: 10, t: 12, b: 40 };
  function draw() {
    NB.clear(s1); NB.clear(s2);
    // left: receptive field profile across space with the spot drawn on top
    const XL = NB.scale(-RMAX, RMAX, m.l, 300 - m.r);
    const prof = (x) => Math.exp(-(x * x) / (2 * sc * sc)) / (sc * sc) - st.k * Math.exp(-(x * x) / (2 * ss * ss)) / (ss * ss);
    const YL = NB.scale(-0.12, 1.05, 300 - m.b, m.t);
    NB.el("rect", { x: XL(-st.r), y: m.t, width: Math.max(0, XL(st.r) - XL(-st.r)), height: 300 - m.t - m.b, fill: NB.c.yellow, "fill-opacity": 0.22 }, s1);
    NB.axes(s1, XL, YL, { xlabel: "position", ylabel: "weight", xticks: 4, yticks: 3, grid: false });
    NB.el("line", { x1: XL(-RMAX), x2: XL(RMAX), y1: YL(0), y2: YL(0), stroke: NB.c.muted, "stroke-width": 1 }, s1);
    const xs = NB.range(-RMAX, RMAX, 241);
    NB.el("path", { d: NB.line(xs.map((x) => [x, Math.exp(-(x * x) / (2 * sc * sc)) / (sc * sc)]), XL, YL), fill: "none", stroke: NB.c.blue, "stroke-width": 1.2, "stroke-opacity": 0.5 }, s1);
    NB.el("path", { d: NB.line(xs.map((x) => [x, -st.k * Math.exp(-(x * x) / (2 * ss * ss)) / (ss * ss)]), XL, YL), fill: "none", stroke: NB.c.orange, "stroke-width": 1.2, "stroke-opacity": 0.7 }, s1);
    NB.el("path", { d: NB.line(xs.map((x) => [x, prof(x)]), XL, YL), fill: "none", stroke: NB.c.fg, "stroke-width": 2 }, s1);
    // right: response vs spot radius
    const XR = NB.scale(0, RMAX, m.l, 420 - m.r), YR = NB.scale(-0.2, 1.05, 300 - m.b, m.t);
    NB.axes(s2, XR, YR, { xlabel: "spot radius", ylabel: "response", xticks: 4, yticks: 5 });
    NB.el("line", { x1: XR(0), x2: XR(RMAX), y1: YR(0), y2: YR(0), stroke: NB.c.muted, "stroke-width": 1 }, s2);
    const rs = NB.range(0, RMAX, 200);
    NB.el("path", { d: NB.line(rs.map((r) => [r, R(r, 0)]), XR, YR), fill: "none", stroke: NB.c.muted, "stroke-width": 1.2, "stroke-dasharray": "4 4" }, s2);
    NB.el("path", { d: NB.line(rs.map((r) => [r, R(r, st.k)]), XR, YR), fill: "none", stroke: NB.c.blue, "stroke-width": 2.5 }, s2);
    let best = 0; rs.forEach((r) => { if (R(r, st.k) > R(best, st.k)) best = r; });
    NB.el("circle", { cx: XR(st.r), cy: YR(R(st.r, st.k)), r: 6, fill: NB.c.blue }, s2);
    NB.el("text", { x: XR(RMAX) - 4, y: YR(R(RMAX, 0)) - 6, class: "nb-tick", "text-anchor": "end" }, s2, "no surround");
    read.innerHTML = `response to this spot <b>${NB.fmt(R(st.r, st.k), 2)}</b>; best spot radius <b>${NB.fmt(best, 1)}</b>` +
      ` (response ${NB.fmt(R(best, st.k), 2)}); a spot covering everything gives <b>${NB.fmt(R(1e3, st.k), 2)}</b>`;
  }
  draw();
});

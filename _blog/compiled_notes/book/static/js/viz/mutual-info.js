/* mutual-info: mutual information of two binary variables. The joint p(x, y) is set by P(X = 1) and the two
 * conditionals P(Y = 1 | X = 0), P(Y = 1 | X = 1), and drawn as a mosaic whose cell areas are p(x, y). On the right,
 * H(X) and H(Y) are bars that overlap by I(X; Y) = H(X) + H(Y) - H(X, Y) (the note's Venn diagram laid flat):
 * the parts outside the overlap are H(X | Y) and H(Y | X), and the two bars together span H(X, Y).
 * The readout also gives D(p(x,y) || p(x)p(y)), which equals I(X; Y). Logs are base 2.
 * params: {px: 0.5, py0: 0.2, py1: 0.7} */
NB.register("mutual-info", function (el, P) {
  const st = { px: P.px ?? 0.5, py0: P.py0 ?? 0.2, py1: P.py1 ?? 0.7 };
  const xlogx = (p) => (p > 0 ? p * Math.log2(p) : 0);
  const Hs = (ps) => -ps.reduce((s, p) => s + xlogx(p), 0);
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "P(X = 1)", min: 0, max: 1, step: 0.01, value: st.px }, (v) => { st.px = v; draw(); });
  NB.slider(ctl, { label: "P(Y = 1 | X = 0)", min: 0, max: 1, step: 0.01, value: st.py0 }, (v) => { st.py0 = v; draw(); });
  NB.slider(ctl, { label: "P(Y = 1 | X = 1)", min: 0, max: 1, step: 0.01, value: st.py1 }, (v) => { st.py1 = v; draw(); });
  const narrow = el.clientWidth > 0 && el.clientWidth < 520;   // phones: mosaic above the bars
  const W = narrow ? 360 : 640, H = narrow ? 500 : 290;
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const mos = narrow ? { x: 66, y: 26, w: 240, h: 240 } : { x: 46, y: 26, w: 220, h: 220 };   // mosaic square
  const bar = narrow ? { x: 34, w: 250, y0: 336, bh: 34 } : { x: 330, w: 250, y0: 70, bh: 34 }; // 1 bit = bar.w / 2

  function draw() {
    NB.clear(svg);
    const px = st.px, p = [[(1 - px) * (1 - st.py0), (1 - px) * st.py0], [px * (1 - st.py1), px * st.py1]]; // p[x][y]
    const pxm = [1 - px, px], pym = [p[0][0] + p[1][0], p[0][1] + p[1][1]];
    const hx = Hs(pxm), hy = Hs(pym), hxy = Hs([p[0][0], p[0][1], p[1][0], p[1][1]]);
    const mi = Math.max(0, hx + hy - hxy);
    let kl = 0;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) if (p[a][b] > 0) kl += p[a][b] * Math.log2(p[a][b] / (pxm[a] * pym[b]));

    // mosaic: column x has width P(X = x); within it, the Y = 1 cell (top) has height P(Y = 1 | X = x)
    const colW = [mos.w * (1 - px), mos.w * px], colX = [mos.x, mos.x + mos.w * (1 - px)], cond = [st.py0, st.py1];
    [0, 1].forEach((x) => {
      if (colW[x] < 0.5) return;
      const h1 = mos.h * cond[x];
      [[1, mos.y, h1], [0, mos.y + h1, mos.h - h1]].forEach(([y, top, hh]) => {
        if (hh < 0.5) return;
        NB.el("rect", { x: colX[x] + 1, y: top + 1, width: Math.max(0, colW[x] - 2), height: Math.max(0, hh - 2), rx: 3,
          fill: y ? NB.c.purple : NB.c.yellow, "fill-opacity": 0.32, stroke: y ? NB.c.purple : NB.c.yellow, "stroke-width": 1.2 }, svg);
        if (colW[x] > 34 && hh > 18)
          NB.el("text", { x: colX[x] + colW[x] / 2, y: top + hh / 2 + 4, "text-anchor": "middle", style: { fontSize: "12px" } }, svg, NB.fmt(p[x][y], 2));
      });
      if (colW[x] > 30) NB.el("text", { x: colX[x] + colW[x] / 2, y: mos.y + mos.h + 16, "text-anchor": "middle", class: "nb-label" }, svg, `X = ${x}`);
    });
    // where the split would be under independence: P(Y = 1) across the whole square
    const yInd = mos.y + mos.h * pym[1];
    NB.el("line", { x1: mos.x - 4, x2: mos.x + mos.w + 4, y1: yInd, y2: yInd, stroke: NB.c.fg, "stroke-dasharray": "4 3", "stroke-width": 1.2 }, svg);
    NB.el("text", { x: mos.x - 6, y: yInd + 4, "text-anchor": "end", class: "nb-tick" }, svg, "P(Y=1)");
    NB.el("text", { x: mos.x + mos.w / 2, y: mos.y - 9, "text-anchor": "middle", class: "nb-label" }, svg, "p(x, y): purple Y = 1, yellow Y = 0");

    // bars: H(X) from 0, H(Y) starting where the overlap begins
    const X = NB.scale(0, 2, bar.x, bar.x + bar.w);
    NB.el("text", { x: X(2) + 10, y: bar.y0 - 22, class: "nb-tick" }, svg, "bits");
    [0, 0.5, 1, 1.5, 2].forEach((t) => {
      NB.el("line", { x1: X(t), x2: X(t), y1: bar.y0 - 18, y2: bar.y0 + 2 * bar.bh + 18, class: "nb-grid" }, svg);
      NB.el("text", { x: X(t), y: bar.y0 - 22, "text-anchor": "middle", class: "nb-tick" }, svg, NB.fmt(t));
    });
    const yStart = hx - mi;
    const rows = [[0, hx, NB.c.blue, "H(X)", bar.y0], [yStart, yStart + hy, NB.c.orange, "H(Y)", bar.y0 + bar.bh + 6]];
    rows.forEach(([a, b, c, lab, y]) => {
      NB.el("rect", { x: X(a), y, width: Math.max(0, X(b) - X(a)), height: bar.bh, rx: 3, fill: c, "fill-opacity": 0.28, stroke: c, "stroke-width": 1.4 }, svg);
      NB.el("text", { x: X(b) + 6, y: y + bar.bh / 2 + 4, class: "nb-label", style: { fill: c, fontWeight: 600 } }, svg, lab);
    });
    // the overlap
    const yb = bar.y0 + 2 * bar.bh + 6;
    if (mi > 0.002) {
      NB.el("rect", { x: X(yStart), y: bar.y0 - 4, width: X(hx) - X(yStart), height: 2 * bar.bh + 14, fill: "none", stroke: NB.c.green, "stroke-width": 2.4, rx: 3 }, svg);
    }
    const seg = (a, b, y, txt, c, always) => {
      if (b - a < 0.002) return;
      NB.el("line", { x1: X(a), x2: X(b), y1: y, y2: y, stroke: c, "stroke-width": 2 }, svg);
      [a, b].forEach((t) => NB.el("line", { x1: X(t), x2: X(t), y1: y - 4, y2: y + 4, stroke: c, "stroke-width": 2 }, svg));
      if (always || X(b) - X(a) > 44) NB.el("text", { x: (X(a) + X(b)) / 2, y: y + (always ? 30 : 16), "text-anchor": "middle", class: "nb-tick", style: { fill: c } }, svg, txt);
    };
    seg(0, yStart, yb + 22, "H(X|Y)", NB.c.blue);
    seg(yStart, hx, yb + 22, "I(X;Y)", NB.c.green, true);
    seg(hx, yStart + hy, yb + 22, "H(Y|X)", NB.c.orange);
    seg(0, hxy, yb + 62, "H(X, Y)", NB.c.fg);

    read.innerHTML = `H(X) = <b>${NB.fmt(hx, 3)}</b> · H(Y) = <b>${NB.fmt(hy, 3)}</b> · H(X,Y) = <b>${NB.fmt(hxy, 3)}</b> bits<br>` +
      `I(X;Y) = H(X) + H(Y) − H(X,Y) = <b>${NB.fmt(mi, 3)}</b> · D(p(x,y) || p(x)p(y)) = <b>${NB.fmt(Math.max(0, kl), 3)}</b>`;
  }
  draw();
});

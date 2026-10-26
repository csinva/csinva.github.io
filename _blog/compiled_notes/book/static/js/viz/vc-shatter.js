/* vc-shatter: does a hypothesis class H shatter a set C of points in the plane? Every one of the 2^|C| labelings
 * is tried; a labeling is realizable if some h in H labels exactly the positives as positive.
 *   half-planes: realizable iff the positives and negatives are strictly linearly separable. The directions w
 *     that separate form an open arc whose ends are normals of p_i - p_j, so testing one direction strictly
 *     between each pair of consecutive critical angles is exact.
 *   axis-aligned rectangles: realizable iff the bounding box of the positives contains no negative.
 * Click empty space to add a point (up to 6), click a point to remove it, or pick a preset.
 * params: {cls: "halfplane", preset: "tri"} */
NB.register("vc-shatter", function (el, P) {
  const PRESETS = {
    tri: [[0.25, 0.3], [0.75, 0.35], [0.5, 0.75]],
    line3: [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]],
    square: [[0.28, 0.28], [0.72, 0.28], [0.72, 0.72], [0.28, 0.72]],
    diamond: [[0.5, 0.18], [0.82, 0.5], [0.5, 0.82], [0.18, 0.5]],
    five: [[0.5, 0.15], [0.85, 0.45], [0.68, 0.85], [0.3, 0.82], [0.15, 0.42]],
  };
  const st = { cls: P.cls || "halfplane", pts: (PRESETS[P.preset || "tri"] || PRESETS.tri).map((p) => p.slice()), sel: -1 };

  function halfplane(pts, lab) {   // returns {w, b} with w.x > b exactly on the positives, or null
    const pos = pts.filter((_, i) => lab[i]), neg = pts.filter((_, i) => !lab[i]);
    if (!pos.length || !neg.length) return { all: pos.length ? 1 : 0 };
    const ang = [0];
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const a = Math.atan2(pts[j][1] - pts[i][1], pts[j][0] - pts[i][0]) + Math.PI / 2;
      ang.push(((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI), ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI));
    }
    ang.sort((a, b) => a - b); ang.push(ang[0] + 2 * Math.PI);
    let best = null;
    for (let i = 0; i + 1 < ang.length; i++) {
      if (ang[i + 1] - ang[i] < 1e-9) continue;
      const t = (ang[i] + ang[i + 1]) / 2, w = [Math.cos(t), Math.sin(t)], pr = (p) => w[0] * p[0] + w[1] * p[1];
      const lo = Math.min(...pos.map(pr)), hi = Math.max(...neg.map(pr));
      if (lo > hi + 1e-9 && (!best || lo - hi > best.gap)) best = { w, b: (lo + hi) / 2, gap: lo - hi };
    }
    return best;
  }
  function rect(pts, lab) {
    const pos = pts.filter((_, i) => lab[i]);
    if (!pos.length) return { all: 0 };
    const r = [Math.min(...pos.map((p) => p[0])), Math.min(...pos.map((p) => p[1])), Math.max(...pos.map((p) => p[0])), Math.max(...pos.map((p) => p[1]))];
    const bad = pts.some((p, i) => !lab[i] && p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3]);
    return bad ? null : { r };
  }

  const ctl = NB.controls(el);
  NB.select(ctl, "class H", [["halfplane", "half-planes (lines)"], ["rect", "axis-aligned rectangles"]], st.cls, (v) => { st.cls = v; draw(); });
  NB.select(ctl, "points C", [["tri", "3 points"], ["line3", "3 points on a line"], ["square", "4: square"], ["diamond", "4: diamond"], ["five", "5 points"], ["custom", "(click to add)"]],
    P.preset || "tri", (v) => { if (PRESETS[v]) st.pts = PRESETS[v].map((p) => p.slice()); st.sel = -1; draw(); });
  NB.button(ctl, "clear", () => { st.pts = []; st.sel = -1; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px", alignItems: "start" } }, el);
  const S = 300, s1 = NB.svg(row, S, S), GW = 420, s2 = NB.svg(row, GW, 300);
  s1.style.cursor = "crosshair"; s1.style.touchAction = "manipulation";
  NB.legend(el, [["positive", NB.c.green], ["negative", NB.c.pink], ["a hypothesis that realizes the labeling", NB.c.blue]]);
  const read = NB.readout(el);
  const X = NB.scale(0, 1, 8, S - 8), Y = NB.scale(0, 1, S - 8, 8);

  s1.addEventListener("click", (e) => {
    const pt = s1.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const q = pt.matrixTransform(s1.getScreenCTM().inverse()), x = X.inv(q.x), y = Y.inv(q.y);
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    const hit = st.pts.findIndex((p) => Math.hypot(X(p[0]) - q.x, Y(p[1]) - q.y) < 12);
    if (hit >= 0) st.pts.splice(hit, 1); else if (st.pts.length < 6) st.pts.push([x, y]);
    st.sel = -1; draw();
  });

  function drawHyp(g, sx, sy, h) {   // a hypothesis inside the box [0,1]^2 drawn with scales sx, sy
    if (!h) return;
    if (h.r) { const pad = 0.04; NB.el("rect", { x: sx(h.r[0] - pad), y: sy(h.r[3] + pad), width: sx(h.r[2] + pad) - sx(h.r[0] - pad), height: sy(h.r[1] - pad) - sy(h.r[3] + pad), fill: NB.c.blue, "fill-opacity": 0.12, stroke: NB.c.blue, "stroke-width": 1.5 }, g); return; }
    if (h.w) {   // half-plane w.x > b: clip the polygon [0,1]^2 to it
      let poly = [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]], out = [];
      const v = (p) => h.w[0] * p[0] + h.w[1] * p[1] - h.b;
      for (let i = 0; i < poly.length; i++) { const a = poly[i], c = poly[(i + 1) % poly.length], va = v(a), vc = v(c);
        if (va > 0) out.push(a);
        if ((va > 0) !== (vc > 0)) { const t = va / (va - vc); out.push([a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1])]); } }
      if (out.length) NB.el("path", { d: out.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("") + "Z", fill: NB.c.blue, "fill-opacity": 0.12, stroke: "none" }, g);
      const d = [-h.w[1], h.w[0]], c0 = [h.w[0] * h.b, h.w[1] * h.b];
      NB.el("line", { x1: sx(c0[0] - 2 * d[0]), y1: sy(c0[1] - 2 * d[1]), x2: sx(c0[0] + 2 * d[0]), y2: sy(c0[1] + 2 * d[1]), stroke: NB.c.blue, "stroke-width": 1.5 }, g);
      return;
    }
    if (h.all) NB.el("rect", { x: sx(-0.02), y: sy(1.02), width: sx(1.02) - sx(-0.02), height: sy(-0.02) - sy(1.02), fill: NB.c.blue, "fill-opacity": 0.12 }, g);
  }

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const pts = st.pts, n = pts.length, test = st.cls === "rect" ? rect : halfplane;
    const labs = Array.from({ length: 1 << n }, (_, m) => pts.map((_, i) => (m >> i) & 1));
    const hyps = labs.map((l) => test(pts, l)), ok = hyps.filter(Boolean).length;
    if (st.sel >= labs.length) st.sel = -1;
    const fail0 = hyps.findIndex((h) => !h), show = st.sel >= 0 ? st.sel : fail0 >= 0 ? fail0 : Math.min(5, labs.length - 1);
    // main view: the chosen labeling (by default the first one H can't realize)
    const clipId = "vcclip" + Math.random().toString(36).slice(2, 8);
    const cp = NB.el("clipPath", { id: clipId }, NB.el("defs", {}, s1)); NB.el("rect", { x: X(0), y: Y(1), width: X(1) - X(0), height: Y(0) - Y(1) }, cp);
    NB.el("rect", { x: X(0), y: Y(1), width: X(1) - X(0), height: Y(0) - Y(1), fill: "none", stroke: NB.c.grid, "stroke-width": 1.5 }, s1);
    const g1 = NB.el("g", { "clip-path": `url(#${clipId})` }, s1);
    if (show >= 0) drawHyp(g1, X, Y, hyps[show]);
    pts.forEach((p, i) => NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 8, fill: show >= 0 ? (labs[show][i] ? NB.c.green : NB.c.pink) : NB.c.fg, stroke: NB.c.bg, "stroke-width": 1.5 }, s1));
    if (show >= 0 && !hyps[show]) NB.el("text", { x: X(0.5), y: Y(0.03), "text-anchor": "middle", class: "nb-label" }, s1, "no h in H gives this labeling");
    if (!n) NB.el("text", { x: X(0.5), y: Y(0.5), "text-anchor": "middle", class: "nb-label" }, s1, "click to add points");
    // grid of all labelings
    const cols = n <= 3 ? 4 : 8, rows = Math.ceil(labs.length / cols), cw = GW / cols, chh = Math.min(cw, 300 / Math.max(rows, 1));
    s2.setAttribute("viewBox", `0 0 ${GW} ${rows * chh}`);
    labs.forEach((l, m) => {
      const cx0 = (m % cols) * cw, cy0 = Math.floor(m / cols) * chh, pad = 6;
      const sx = NB.scale(0, 1, cx0 + pad, cx0 + cw - pad), sy = NB.scale(0, 1, cy0 + chh - pad, cy0 + pad);
      const cell = NB.el("g", { style: { cursor: "pointer" } }, s2);
      NB.el("rect", { x: cx0 + 2, y: cy0 + 2, width: cw - 4, height: chh - 4, rx: 5, fill: m === show ? NB.c.grid : NB.c.bg, stroke: hyps[m] ? NB.c.grid : NB.c.pink, "stroke-width": hyps[m] ? 1 : 2 }, cell);
      const cid = clipId + "c" + m, c2 = NB.el("clipPath", { id: cid }, NB.el("defs", {}, cell));
      NB.el("rect", { x: cx0 + 3, y: cy0 + 3, width: cw - 6, height: chh - 6 }, c2);
      drawHyp(NB.el("g", { "clip-path": `url(#${cid})` }, cell), sx, sy, hyps[m]);
      pts.forEach((p, i) => NB.el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: Math.max(2.2, chh / 16), fill: l[i] ? NB.c.green : NB.c.pink }, cell));
      if (!hyps[m]) NB.el("text", { x: cx0 + cw - 8, y: cy0 + 15, "text-anchor": "end", fill: NB.c.pink, style: { fontSize: "13px", fontWeight: 700 } }, cell, "×");
      cell.addEventListener("click", () => { st.sel = m; draw(); });
    });
    const shat = n > 0 && ok === labs.length;
    read.innerHTML = n ? `${n} points: H realizes <b>${ok}</b> of the 2<sup>${n}</sup> = ${labs.length} labelings, so it ${shat ? "<b>shatters</b> this set" : "does <b>not</b> shatter this set"}. ` +
      `Click a labeling to see it on the left${show >= 0 && !hyps[show] && st.sel < 0 ? " (shown: the first one that fails)" : ""}.` : "Add points by clicking the square.";
  }
  draw();
});

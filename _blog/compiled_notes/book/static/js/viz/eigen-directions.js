/* eigen-directions: a 2x2 matrix A = [[a, b], [c, d]], a vector v (drag it) and Av. The lines through the origin
 * along the real eigenvectors are drawn in blue (the note's color for Q, the eigenvectors) with their eigenvalues
 * in orange (the color for Lambda). On those lines Av stays on the line through v: Av = lambda v.
 * Eigenvalues from the characteristic polynomial lambda^2 - tr(A) lambda + det(A) = 0; checked against
 * numpy.linalg.eig for every preset.
 * params: {preset: "general" | "symmetric" | "rotation" | "shear", angle: 60} */
NB.register("eigen-directions", function (el, P) {
  const presets = {
    general: [[1.5, 1], [0.5, 1]],
    symmetric: [[2, 1], [1, 2]],
    rotation: [[0, -1], [1, 0]],
    shear: [[1, 1], [0, 1]],
  };
  const st = { A: presets[P.preset || "general"].map((r) => r.slice()), v: [Math.cos(((P.angle ?? 60) * Math.PI) / 180), Math.sin(((P.angle ?? 60) * Math.PI) / 180)].map((x) => 1.3 * x) };
  const ctl = NB.controls(el);
  const sel = NB.select(ctl, "preset", [["general", "general"], ["symmetric", "symmetric"], ["rotation", "rotation by 90°"], ["shear", "shear (defective)"], ["custom", "custom"]], P.preset || "general", (k) => {
    if (k === "custom") return;
    st.A = presets[k].map((r) => r.slice()); sliders.forEach((s, i) => s.set(st.A[i >> 1][i & 1])); draw();
  });
  const sliders = [["a", 0, 0], ["b", 0, 1], ["c", 1, 0], ["d", 1, 1]].map(([lab, i, j]) =>
    NB.slider(ctl, { label: lab, min: -2, max: 2, step: 0.1, value: st.A[i][j], format: (v) => v.toFixed(1) }, (v) => { st.A[i][j] = v; sel.value = "custom"; draw(); }));
  const W = 640, H = 380, L = 3, S = H - 20, ox = 10;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["v (drag the tip)", NB.c.pink], ["Av", NB.c.purple], ["line through v", NB.c.muted, true], ["eigenvector directions", NB.c.blue]]);
  const read = NB.readout(el);
  const X = NB.scale(-L, L, ox, ox + S), Y = NB.scale(-L, L, 10 + S, 10);
  const cp = NB.el("clipPath", { id: "ed" + Math.random().toString(36).slice(2, 8) }, NB.el("defs", {}, svg));
  NB.el("rect", { x: ox, y: 10, width: S, height: S }, cp);
  const gP = NB.el("g", { "clip-path": `url(#${cp.id})` }, svg), gT = NB.el("g", {}, svg);
  const hit = NB.el("rect", { x: ox, y: 10, width: S, height: S, fill: "transparent", style: { cursor: "pointer", touchAction: "none" } }, svg);
  svg.insertBefore(hit, gT);   // matrix text stays on top but doesn't catch the pointer
  gT.setAttribute("pointer-events", "none");

  // real eigenpairs of a 2x2 matrix
  function eig(A) {
    const [[a, b], [c, d]] = A, tr = a + d, det = a * d - b * c, disc = (tr * tr) / 4 - det;
    if (disc < -1e-12) return { complex: true, re: tr / 2, im: Math.sqrt(-disc), tr, det };
    const r = disc < 1e-10 ? 0 : Math.sqrt(disc), lams = r < 1e-9 ? [tr / 2] : [tr / 2 + r, tr / 2 - r];
    if (r < 1e-9 && Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9) return { all: true, lams, tr, det };   // A = lambda I
    // rows of A - lambda I: pick the one that is not zero to get a direction
    const vec2 = (l) => {
      const r1 = [a - l, b], r2 = [c, d - l], row = Math.hypot(...r1) > Math.hypot(...r2) ? r1 : r2;
      if (Math.hypot(...row) < 1e-9) return [1, 0];
      const n = Math.hypot(...row); return [-row[1] / n, row[0] / n];
    };
    return { pairs: lams.map((l) => ({ l, u: vec2(l) })), lams, tr, det, defective: r < 1e-9 };
  }
  const mv = (A, v) => [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]];
  const arrow = (g, x1, y1, color, w) => {
    const px = [X(0), Y(0), X(x1), Y(y1)], ang = Math.atan2(px[3] - px[1], px[2] - px[0]), len = Math.hypot(px[2] - px[0], px[3] - px[1]);
    if (len < 1) return;
    const hl = Math.min(12, len * 0.5);
    NB.el("line", { x1: px[0], y1: px[1], x2: px[2] - Math.cos(ang) * hl * 0.8, y2: px[3] - Math.sin(ang) * hl * 0.8, stroke: color, "stroke-width": w || 3 }, g);
    NB.el("polygon", { points: [[px[2], px[3]], [px[2] - hl * Math.cos(ang - 0.4), px[3] - hl * Math.sin(ang - 0.4)], [px[2] - hl * Math.cos(ang + 0.4), px[3] - hl * Math.sin(ang + 0.4)]].map((p) => p.join(",")).join(" "), fill: color }, g);
  };
  const clampBox = (v, r) => { const k = Math.max(Math.abs(v[0]), Math.abs(v[1])); return k > r ? [(v[0] * r) / k, (v[1] * r) / k] : v; };
  const clampLen = (v, r) => { const n = Math.hypot(...v); return n > r ? [(v[0] * r) / n, (v[1] * r) / n] : v; };

  let dragging = false;
  const move = (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    let v = clampLen([X.inv(p.x), Y.inv(p.y)], L * 0.95);
    if (Math.hypot(...v) < 0.15) return;
    const E = eig(st.A);
    (E.pairs || []).forEach(({ u }) => {   // snap onto an eigen direction within 3 degrees
      const n = Math.hypot(...v), c = (v[0] * u[0] + v[1] * u[1]) / n;
      if (Math.abs(c) > Math.cos((3 * Math.PI) / 180)) v = [Math.sign(c) * n * u[0], Math.sign(c) * n * u[1]];
    });
    st.v = v; draw();
  };
  hit.addEventListener("pointerdown", (e) => { dragging = true; hit.setPointerCapture(e.pointerId); move(e); });
  hit.addEventListener("pointermove", (e) => { if (dragging) move(e); });
  hit.addEventListener("pointerup", () => { dragging = false; });

  function draw() {
    NB.clear(gP); NB.clear(gT);
    const A = st.A, E = eig(A), v = st.v, Av = mv(A, v);
    NB.el("rect", { x: ox, y: 10, width: S, height: S, fill: "none", stroke: NB.c.grid }, gP);
    for (let k = -3; k <= 3; k++) {
      NB.el("line", { x1: X(k), x2: X(k), y1: Y(-L), y2: Y(L), class: "nb-grid" }, gP);
      NB.el("line", { y1: Y(k), y2: Y(k), x1: X(-L), x2: X(L), class: "nb-grid" }, gP);
    }
    NB.el("line", { x1: X(-L), x2: X(L), y1: Y(0), y2: Y(0), class: "nb-axis" }, gP);
    NB.el("line", { x1: X(0), x2: X(0), y1: Y(-L), y2: Y(L), class: "nb-axis" }, gP);
    const ext = (u, s) => [u[0] * s, u[1] * s];
    const onEig = (E.pairs || []).find(({ u }) => Math.abs(v[0] * u[1] - v[1] * u[0]) / Math.hypot(...v) < 1e-6);
    (E.pairs || []).forEach(({ l, u }, i) => {
      const [p, q] = [ext(u, 2 * L), ext(u, -2 * L)];
      NB.el("line", { x1: X(p[0]), y1: Y(p[1]), x2: X(q[0]), y2: Y(q[1]), stroke: NB.c.blue, "stroke-width": onEig && onEig.u === u ? 5 : 2.5, "stroke-opacity": onEig && onEig.u === u ? 0.5 : 0.8 }, gP);
      // eigenvalue label near the lower end of the line, just above it (v starts in the upper half)
      const sgn = u[1] > 1e-9 ? -1 : 1, t = 0.93 * Math.min(L / Math.max(Math.abs(u[0]), 1e-9), L / Math.max(Math.abs(u[1]), 1e-9));
      const px = sgn * u[0] * t, py = sgn * u[1] * t, right = px >= -1e-9, flat = Math.abs(u[1]) < 0.2;
      NB.el("text", { x: X(px) + (right ? -6 : 6), y: Y(py) - 8, "text-anchor": right ? "end" : "start", style: { fontSize: "14px", fill: NB.c.orange, fontWeight: 600 } }, gT, `λ${E.pairs.length > 1 ? "₁₂"[i] : ""} = ${NB.fmt(l, 2)}`);
    });
    // the line through v: if Av lies on it, v is an eigenvector
    const vn = Math.hypot(...v), vu = [v[0] / vn, v[1] / vn];
    NB.el("line", { x1: X(vu[0] * 2 * L), y1: Y(vu[1] * 2 * L), x2: X(-vu[0] * 2 * L), y2: Y(-vu[1] * 2 * L), stroke: NB.c.muted, "stroke-dasharray": "5 5", "stroke-width": 1.4 }, gP);
    const Avc = clampBox(Av, L * 0.97);   // long Av is drawn shortened to the edge of the plot
    arrow(gP, Avc[0], Avc[1], NB.c.purple, 5);   // wider than v, so it still shows when Av lies on top of v
    arrow(gP, v[0], v[1], NB.c.pink, 3);
    NB.el("circle", { cx: X(v[0]), cy: Y(v[1]), r: 9, fill: NB.c.pink, "fill-opacity": 0.25, stroke: NB.c.pink }, gP);
    // labels on opposite sides of the line through v, so they don't collide when Av is along v
    const nrm = [-vu[1], vu[0]], lab = (w, side, t, color) => NB.el("text", { x: Math.min(ox + S - 12, Math.max(ox + 12, X(w[0] + side * 0.3 * nrm[0]))),
      y: Math.min(S, Math.max(26, Y(w[1] + side * 0.3 * nrm[1]) + 5)), "text-anchor": "middle", style: { fontSize: "14px", fill: color, fontWeight: 600 } }, gT, t);
    lab(v, 1, "v", NB.c.pink);
    if (Math.hypot(...Avc) > 0.05) lab(Avc, -1, "Av", NB.c.purple);

    // right side: the matrix and the characteristic polynomial
    const tx = ox + S + 28, f = (x) => NB.fmt(x, 2);
    NB.el("text", { x: tx, y: 40, class: "nb-label" }, gT, "A =");
    [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(([i, j]) => NB.el("text", { x: tx + 64 + j * 50, y: 30 + i * 22, "text-anchor": "middle", style: { fontSize: "15px" } }, gT, f(A[i][j])));
    NB.el("path", { d: `M${tx + 34},14v42M${tx + 34},14h4M${tx + 34},56h4M${tx + 140},14v42M${tx + 140},14h-4M${tx + 140},56h-4`, stroke: NB.c.fg, fill: "none" }, gT);
    const lines = [`tr A = ${f(E.tr)},  det A = ${f(E.det)}`, `λ² − (tr A) λ + det A = 0`];
    if (E.complex) lines.push(`λ = ${f(E.re)} ± ${f(E.im)}i (complex)`, "no real eigenvectors:", "every v gets turned");
    else if (E.all) lines.push(`λ = ${f(E.lams[0])} (twice), A = λI:`, "every v is an eigenvector");
    else if (E.defective) lines.push(`λ = ${f(E.lams[0])} (twice), but only`, "one eigenvector direction");
    else lines.push(`λ₁ = ${f(E.lams[0])},  λ₂ = ${f(E.lams[1])}`, `λ₁λ₂ = det A,  λ₁ + λ₂ = tr A`);
    lines.forEach((s, k) => NB.el("text", { x: tx, y: 94 + k * 25, style: { fontSize: "15px", fill: k >= 2 ? NB.c.orange : NB.c.fg } }, gT, s));
    if (E.pairs && E.pairs.length === 2) {
      const [u1, u2] = E.pairs.map((p) => p.u), ang = (Math.acos(Math.min(1, Math.abs(u1[0] * u2[0] + u1[1] * u2[1]))) * 180) / Math.PI;
      NB.el("text", { x: tx, y: 94 + lines.length * 25, style: { fontSize: "15px", fill: NB.c.blue } }, gT, `angle between eigenvectors: ${ang.toFixed(0)}°`);
    }

    const cross = v[0] * Av[1] - v[1] * Av[0], dot = v[0] * Av[0] + v[1] * Av[1];
    const turn = Math.abs((Math.atan2(cross, dot) * 180) / Math.PI);
    let msg = `v = (${f(v[0])}, ${f(v[1])}), Av = (${f(Av[0])}, ${f(Av[1])}). `;
    if (E.all) msg += `A is a multiple of I, so every direction is kept: Av = <b>${f(E.lams[0])}</b> v.`;
    else if (onEig) msg += `Av is on the line through v: <b>Av = ${f(onEig.l)} v</b>, so v is an eigenvector with eigenvalue ${f(onEig.l)}` +
      (onEig.l < 0 ? " (it flips v)." : Math.abs(onEig.l) < 1e-9 ? " (A sends it to 0, so A is singular)." : ".");
    else msg += `Av is turned <b>${turn.toFixed(0)}°</b> away from the line through v, so v is not an eigenvector` +
      (E.complex ? ". With complex eigenvalues no direction is kept." : `. Drag v onto a blue line.`);
    read.innerHTML = msg;
  }
  draw();
});

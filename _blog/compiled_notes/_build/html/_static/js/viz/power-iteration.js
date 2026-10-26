/* power-iteration: u_{k+1} = A u_k for a 2x2 matrix A = Q Λ Q^{-1} with fixed (non-orthogonal) eigenvectors
 * q1, q2 (columns of Q, blue as in the note) and eigenvalues λ1, λ2 on sliders (orange, the note's color for Λ).
 * Left: the iterates u_0, u_1, ..., u_k in the plane. Since u_k = c1 λ1^k q1 + c2 λ2^k q2, the iterates shrink to 0
 * when both |λ| < 1 and run off when one |λ| > 1. Right: the angle between u_k and the line of the dominant
 * eigenvector (larger |λ|), log scale, with the rate |λ2/λ1|^k (scaled to start at the same angle) dashed: the
 * direction converges at that rate whatever the size of u_k, which is the power method v = lim A^k v0 / |A^k v0|.
 * Iterates from repeated multiplication by A; checked against numpy (A = Q diag(λ) inv(Q), np.linalg.matrix_power,
 * angle via arctan2(|cross|, |dot|)) for (λ1, λ2) in {(1.2, 0.6), (0.9, 0.5), (0.8, -0.95), (1, 1)}.
 * params: {l1: 1.2, l2: 0.6, k: 8} */
NB.register("power-iteration", function (el, P) {
  const deg = Math.PI / 180;
  const q1 = [Math.cos(20 * deg), Math.sin(20 * deg)], q2 = [Math.cos(115 * deg), Math.sin(115 * deg)];
  const u0 = [0.3, 1.0];
  const st = { l1: P.l1 ?? 1.2, l2: P.l2 ?? 0.6, k: P.k ?? 8 };
  const ctl = NB.controls(el);
  const fmtL = (v) => v.toFixed(2);
  NB.slider(ctl, { label: "λ₁", min: -1.5, max: 1.5, step: 0.05, value: st.l1, format: fmtL }, (v) => { st.l1 = v; draw(); });
  NB.slider(ctl, { label: "λ₂", min: -1.5, max: 1.5, step: 0.05, value: st.l2, format: fmtL }, (v) => { st.l2 = v; draw(); });
  NB.slider(ctl, { label: "steps k", min: 0, max: 25, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; draw(); });
  const W = 640, H = 320;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["eigenvectors q₁, q₂", NB.c.blue], ["u₀, u₁, …, uₖ", NB.c.purple], ["angle to dominant eigenvector", NB.c.pink], ["rate (smaller |λ| / larger |λ|)ᵏ", NB.c.muted, true]]);
  const read = NB.readout(el);

  function matrix() {               // A = Q diag(l1, l2) Q^{-1}
    const [a, c] = q1, [b, d] = q2, det = a * d - b * c;
    const Qi = [[d / det, -b / det], [-c / det, a / det]];
    const QL = [[a * st.l1, b * st.l2], [c * st.l1, d * st.l2]];
    return [[QL[0][0] * Qi[0][0] + QL[0][1] * Qi[1][0], QL[0][0] * Qi[0][1] + QL[0][1] * Qi[1][1]],
            [QL[1][0] * Qi[0][0] + QL[1][1] * Qi[1][0], QL[1][0] * Qi[0][1] + QL[1][1] * Qi[1][1]]];
  }
  const lineAngle = (u, q) => Math.atan2(Math.abs(u[0] * q[1] - u[1] * q[0]), Math.abs(u[0] * q[0] + u[1] * q[1])) / deg;

  function draw() {
    NB.clear(svg);
    const A = matrix(), K = 25, us = [u0];
    for (let k = 1; k <= K; k++) { const u = us[k - 1]; us.push([A[0][0] * u[0] + A[0][1] * u[1], A[1][0] * u[0] + A[1][1] * u[1]]); }
    // ---- left: the plane
    const L = 3, S = 296, ox = 8, oy = 12, X = NB.scale(-L, L, ox, ox + S), Y = NB.scale(-L, L, oy + S, oy);
    const cp = NB.el("clipPath", { id: "pi" + Math.random().toString(36).slice(2, 8) }, NB.el("defs", {}, svg));
    NB.el("rect", { x: ox, y: oy, width: S, height: S }, cp);
    NB.el("rect", { x: ox, y: oy, width: S, height: S, fill: "none", stroke: NB.c.grid }, svg);
    const g = NB.el("g", { "clip-path": `url(#${cp.id})` }, svg);
    NB.el("line", { x1: X(-L), x2: X(L), y1: Y(0), y2: Y(0), class: "nb-grid" }, g);
    NB.el("line", { x1: X(0), x2: X(0), y1: Y(-L), y2: Y(L), class: "nb-grid" }, g);
    const dom = Math.abs(st.l1) >= Math.abs(st.l2) ? 1 : 2, tie = Math.abs(Math.abs(st.l1) - Math.abs(st.l2)) < 1e-9;
    [[q1, st.l1, "q₁", 1], [q2, st.l2, "q₂", 2]].forEach(([q, l, name, i]) => {
      const big = !tie && i === dom;
      NB.el("line", { x1: X(-4 * q[0]), y1: Y(-4 * q[1]), x2: X(4 * q[0]), y2: Y(4 * q[1]), stroke: NB.c.blue, "stroke-width": big ? 2.5 : 1.2, "stroke-opacity": big ? 0.9 : 0.55 }, g);
      const tx = X(-2.2 * q[0]) + 6, ty = Y(-2.2 * q[1]) + 16;
      const t = NB.el("text", { x: tx, y: ty, "font-size": 13, fill: NB.c.blue, "text-anchor": "start" }, g, name + ": ");
      NB.el("tspan", { fill: NB.c.orange }, t, `λ${i === 1 ? "₁" : "₂"} = ` + l.toFixed(2));
    });
    const pts = us.slice(0, st.k + 1);
    NB.el("path", { d: pts.map((u, i) => (i ? "L" : "M") + X(u[0]).toFixed(1) + "," + Y(u[1]).toFixed(1)).join(""), fill: "none", stroke: NB.c.purple, "stroke-width": 1.3, "stroke-opacity": 0.6 }, g);
    pts.forEach((u, i) => NB.el("circle", { cx: X(u[0]), cy: Y(u[1]), r: i === st.k ? 5 : 3.2, fill: NB.c.purple, "fill-opacity": i === 0 ? 0.5 : 1 }, g));
    NB.el("text", { x: X(u0[0]) + 7, y: Y(u0[1]) + 4, "font-size": 12, fill: NB.c.purple }, g, "u₀");
    // ---- right: angle to the dominant eigenvector, log scale
    const qd = dom === 1 ? q1 : q2;
    const ang = us.map((u) => lineAngle(u, qd));
    const x0 = 380, x1 = 630, y0 = oy + S - 22, y1 = oy;
    const KX = NB.scale(0, K, x0, x1), lo = -6, hi = Math.log10(90), LY = NB.scale(lo, hi, y0, y1);
    NB.axes(svg, KX, LY, { xlabel: "k", ylabel: "angle (degrees)", xticks: 5, yticks: 6, yfmt: (t) => (t === 0 ? "1" : "10" + String(t).split("").map((ch) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹".charAt("-0123456789".indexOf(ch))).join("")) });
    const lg = (a) => Math.max(lo, Math.log10(Math.max(a, 1e-12)));
    const finite = us.map((u) => isFinite(u[0]) && isFinite(u[1]) && Math.hypot(u[0], u[1]) > 1e-300);
    const path = ang.map((a, k) => (finite[k] ? [k, lg(a)] : null)).filter(Boolean);
    NB.el("path", { d: NB.line(path, KX, LY), fill: "none", stroke: NB.c.pink, "stroke-width": 2 }, svg);
    if (!tie && st.l1 !== 0 && st.l2 !== 0) {
      const r = Math.min(Math.abs(st.l1), Math.abs(st.l2)) / Math.max(Math.abs(st.l1), Math.abs(st.l2));
      const ref = Array.from({ length: K + 1 }, (_, k) => [k, lg(ang[0] * Math.pow(r, k))]);
      NB.el("path", { d: NB.line(ref, KX, LY), fill: "none", stroke: NB.c.muted, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    }
    if (finite[st.k]) NB.el("circle", { cx: KX(st.k), cy: LY(lg(ang[st.k])), r: 4.5, fill: NB.c.pink }, svg);
    // ---- readout
    const uk = us[st.k], nk = Math.hypot(uk[0], uk[1]), mx = Math.max(Math.abs(st.l1), Math.abs(st.l2));
    const stab = mx < 1 - 1e-9 ? "both |λ| &lt; 1: <b>stable</b>, uₖ → 0" : Math.abs(mx - 1) < 1e-9 ? "largest |λ| = 1: <b>neutrally stable</b>" : "some |λ| &gt; 1: <b>unstable</b>, |uₖ| grows";
    const dir = tie ? "|λ₁| = |λ₂|: no dominant eigenvalue, so the direction does not settle"
      : `angle between u<sub>${st.k}</sub> and q${dom === 1 ? "₁" : "₂"}: <b>${NB.fmt(ang[st.k], 3)}°</b> (smaller |λ| / larger |λ| = ${(Math.min(Math.abs(st.l1), Math.abs(st.l2)) / mx).toFixed(2)})`;
    read.innerHTML = `|u<sub>${st.k}</sub>| = <b>${NB.fmt(nk, 3)}</b>; ${stab}. ${dir}.`;
  }
  draw();
});

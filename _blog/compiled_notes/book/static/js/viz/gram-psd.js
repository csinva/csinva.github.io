/* gram-psd: is K a valid kernel? Build the n x n Gram matrix K_ij = K(x_i, x_j) on n points of the real line
 * (sorted, so nearby points sit next to each other in the matrix) and compute its eigenvalues with the cyclic
 * Jacobi method. A valid kernel gives a positive semi-definite matrix (no negative eigenvalue) for every set of
 * points; one negative eigenvalue on one set is enough to rule a function out.
 * params: {kernel: "rbf", r: 1, n: 8, seed: 3} */
NB.register("gram-psd", function (el, P) {
  const st = { kernel: P.kernel || "rbf", r: P.r ?? 1, n: P.n || 8, seed: P.seed || 3 };
  const KERNELS = {
    linear: { label: "linear  xz", f: (x, z) => x * z },
    poly: { label: "polynomial  (1 + xz)²", f: (x, z) => (1 + x * z) ** 2 },
    rbf: { label: "RBF  exp(−r(x − z)²)", f: (x, z, r) => Math.exp(-r * (x - z) ** 2) },
    sum: { label: "sum  RBF + polynomial", f: (x, z, r) => Math.exp(-r * (x - z) ** 2) + (1 + x * z) ** 2 },
    prod: { label: "product  RBF × polynomial", f: (x, z, r) => Math.exp(-r * (x - z) ** 2) * (1 + x * z) ** 2 },
    negsq: { label: "not a kernel: −r(x − z)²", f: (x, z, r) => -r * (x - z) ** 2 },
    box: { label: "not a kernel: 1 if |x − z| < 1/√r", f: (x, z, r) => (Math.abs(x - z) < 1 / Math.sqrt(r) ? 1 : 0) },
  };
  function eig(A) {   // cyclic Jacobi for a symmetric matrix; returns eigenvalues sorted descending
    const n = A.length, a = A.map((r) => r.slice());
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      }
    }
    return a.map((r, i) => r[i]).sort((x, y) => y - x);
  }
  const pts = () => { const r = NB.rng(7919 * st.seed + 5); for (let i = 0; i < 5; i++) r();
    return Array.from({ length: st.n }, () => -2 + 4 * r()).sort((a, b) => a - b); };

  const ctl = NB.controls(el);
  NB.select(ctl, "K(x, z)", Object.keys(KERNELS).map((k) => [k, KERNELS[k].label]), st.kernel, (v) => { st.kernel = v; draw(); });
  NB.slider(ctl, { label: "r", min: 0.1, max: 10, step: 0.1, value: st.r, format: (v) => v.toFixed(1) }, (v) => { st.r = v; draw(); });
  NB.slider(ctl, { label: "points n", min: 2, max: 14, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  NB.button(ctl, "new points", () => { st.seed++; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px", alignItems: "start" } }, el);
  const S = 320, s1 = NB.svg(row, S, S + 34), W2 = 400, H2 = 354, s2 = NB.svg(row, W2, H2);
  NB.legend(el, [["K_ij > 0", NB.c.blue], ["K_ij < 0", NB.c.orange], ["eigenvalue ≥ 0", NB.c.green], ["eigenvalue < 0", NB.c.pink]]);
  const read = NB.readout(el);

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const xs = pts(), n = xs.length, K = KERNELS[st.kernel];
    const G = xs.map((x) => xs.map((z) => K.f(x, z, st.r))), ev = eig(G);
    // the points on a line, then the Gram matrix as a heatmap
    const L = NB.scale(-2, 2, 30, S - 10);
    NB.el("line", { x1: L(-2), x2: L(2), y1: 14, y2: 14, class: "nb-axis" }, s1);
    xs.forEach((x) => NB.el("circle", { cx: L(x), cy: 14, r: 3.5, fill: NB.c.fg }, s1));
    NB.el("text", { x: L(-2), y: 32, class: "nb-tick" }, s1, "−2"); NB.el("text", { x: L(2), y: 32, "text-anchor": "end", class: "nb-tick" }, s1, "2");
    const top = 40, cs = (S - 40) / n, mx = Math.max(...G.flat().map(Math.abs), 1e-12);
    G.forEach((rw, i) => rw.forEach((v, j) => NB.el("rect", { x: 30 + j * cs, y: top + i * cs, width: cs - 0.6, height: cs - 0.6,
      fill: v >= 0 ? NB.c.blue : NB.c.orange, "fill-opacity": (0.06 + 0.94 * Math.abs(v) / mx).toFixed(3) }, s1)));
    NB.el("text", { x: 22, y: top + (S - 40) / 2, "text-anchor": "middle", class: "nb-label", transform: `rotate(-90 22 ${top + (S - 40) / 2})` }, s1, "Gram matrix K_n, row i, column j");
    // eigenvalues
    const lo = Math.min(0, ...ev), hi = Math.max(0, ...ev), pad = 0.08 * (hi - lo || 1);
    const m = { l: 52, r: 10, t: 14, b: 38 }, X = NB.scale(0.4, n + 0.6, m.l, W2 - m.r), Y = NB.scale(lo - (lo < 0 ? pad : 0), hi + pad, H2 - m.b, m.t);
    NB.axes(s2, X, Y, { xlabel: "eigenvalues, largest first", ylabel: "eigenvalue", xticks: Math.min(n, 7), yticks: 5, grid: true });
    NB.el("line", { x1: X(0.4), x2: X(n + 0.6), y1: Y(0), y2: Y(0), stroke: NB.c.fg, "stroke-width": 1 }, s2);
    const bw = Math.min(26, (X(2) - X(1)) * 0.7), tol = 1e-9 * Math.max(1, Math.abs(hi));
    ev.forEach((v, i) => NB.el("rect", { x: X(i + 1) - bw / 2, y: Math.min(Y(v), Y(0)), width: bw, height: Math.max(1, Math.abs(Y(v) - Y(0))), fill: v < -tol ? NB.c.pink : NB.c.green }, s2));
    const negs = ev.filter((v) => v < -tol);
    const show = (v) => (Math.abs(v) < tol ? "0" : NB.fmt(v, 3));
    read.innerHTML = `smallest eigenvalue <b>${show(ev[n - 1])}</b>, largest <b>${NB.fmt(ev[0], 3)}</b> · ` +
      (negs.length ? `<b>${negs.length}</b> negative, so K_n is not PSD and this K is not a valid kernel`
                   : `none negative: K_n is PSD on these points`);
  }
  draw();
});

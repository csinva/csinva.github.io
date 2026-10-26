/* spectral-rings: spectral clustering of two noisy rings with an RBF kernel K(x, z) = exp(-r ||x - z||²).
 * Left: points split by the sign of the second generalized eigenvector of L_n = D_n - K_n (normalized cut).
 * Right: the kernel matrix K_n with points ordered inner ring first, each ring by angle.
 * params: {r: 3} */
NB.register("spectral-rings", function (el, P) {
  const rng = NB.rng(21), pts = [], nIn = 25, nOut = 55;
  [[nIn, 0.8], [nOut, 2.8]].forEach(([n, R]) => { for (let i = 0; i < n; i++) { const t = (2 * Math.PI * (i + 0.6 * rng())) / n; pts.push([R * Math.cos(t) + 0.15 * rng.normal(), R * Math.sin(t) + 0.15 * rng.normal()]); } });
  const N = pts.length, D2 = pts.map((a) => pts.map((b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));
  const st = { r: P.r ?? 3 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "kernel parameter r", min: Math.log10(0.1), max: Math.log10(100), step: 0.02, value: Math.log10(st.r), format: (v) => NB.fmt(10 ** v, 2) }, (v) => { st.r = 10 ** v; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "14px" } }, el);
  const S = 340, s1 = NB.svg(row, S, S), s2 = NB.svg(row, S, S);
  NB.legend(el, [["cluster A (eigenvector > 0)", NB.c.blue], ["cluster B (eigenvector < 0)", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(-3.4, 3.4, 10, S - 10), Y = NB.scale(-3.4, 3.4, S - 10, 10);

  // cyclic Jacobi eigen-decomposition of a symmetric matrix; returns {vals, vecs (columns)}
  function eigh(A0) {
    const n = A0.length, A = A0.map((r) => r.slice()), V = A.map((_, i) => A.map((_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 30; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    return { vals: A.map((r, i) => r[i]), vecs: V };
  }

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const K = D2.map((r) => r.map((d) => Math.exp(-st.r * d))), deg = K.map((r) => r.reduce((a, b) => a + b, 0));
    // L v = λ D v  <=>  (D^-1/2 K D^-1/2) u = (1 - λ) u with v = D^-1/2 u
    const Mn = K.map((r, i) => r.map((k, j) => k / Math.sqrt(deg[i] * deg[j])));
    const { vals, vecs } = eigh(Mn), order = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]);
    const u = order[1], v = pts.map((_, i) => vecs[i][u] / Math.sqrt(deg[i]));
    const lab = v.map((x) => x > 0), sizeA = lab.filter((x) => x).length;
    const ringOk = Math.max(lab.filter((x, i) => x === i < nIn).length, lab.filter((x, i) => x !== i < nIn).length) / N;
    // points
    pts.forEach((p, i) => NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: lab[i] ? NB.c.blue : NB.c.orange, "fill-opacity": 0.85 }, s1));
    // kernel matrix
    const c = (S - 20) / N;
    NB.el("rect", { x: 10, y: 10, width: S - 20, height: S - 20, fill: "none", stroke: NB.c.grid }, s2);
    const g = NB.el("g", {}, s2);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (K[i][j] > 0.02)
      NB.el("rect", { x: 10 + j * c, y: 10 + i * c, width: c + 0.3, height: c + 0.3, fill: NB.c.fg, "fill-opacity": K[i][j].toFixed(2) }, g);
    NB.el("line", { x1: 10 + nIn * c, x2: 10 + nIn * c, y1: 10, y2: S - 10, stroke: NB.c.pink, "stroke-dasharray": "3 3" }, s2);
    NB.el("line", { y1: 10 + nIn * c, y2: 10 + nIn * c, x1: 10, x2: S - 10, stroke: NB.c.pink, "stroke-dasharray": "3 3" }, s2);
    const lam = order.slice(1, 3).map((k) => Math.max(0, 1 - vals[k]));
    const verdict = ringOk > 0.95 ? "the cut separates the two rings"
      : lam[1] < 1e-3 ? "several eigenvalues are near 0: the kernel graph has broken into pieces, so the second eigenvector is not unique and the split is arbitrary"
      : "the kernel is too wide: the cut slices across both rings";
    read.innerHTML = `r = ${NB.fmt(st.r, 2)} · second and third smallest eigenvalues of L: <b>${lam.map((l) => (l < 1e-6 ? "≈ 0" : NB.fmt(l, 4))).join(", ")}</b> · clusters of ${sizeA} and ${N - sizeA} points, <b>${Math.round(ringOk * 100)}%</b> agree with the rings: ${verdict}`;
  }
  draw();
});

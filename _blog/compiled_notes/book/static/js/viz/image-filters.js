/* image-filters: linear filtering of a small grayscale image. Each output pixel is the dot product of the
 * kernel with the input patch around it (cross-correlation, as CNNs compute it; edges are padded by repeating
 * the border pixel). Smoothing kernels are nonnegative and sum to 1; derivative kernels sum to 0.
 * Pick an output pixel to see the patch it came from and the sum.
 * params: {kernel: "sobelx", image: "shapes", sigma: 0.8, row: 8, col: 4} */
NB.register("image-filters", function (el, P) {
  const n = 24, cs = 9, IW = n * cs, W = 640, top = 24, H = top + IW + 26;
  const st = { kernel: P.kernel || "sobelx", image: P.image || "shapes", sigma: P.sigma ?? 0.8, sel: [P.row ?? 8, P.col ?? 4] };

  // ---- images (values in [0, 1]) ----
  function makeImage(kind) {
    const I = Array.from({ length: n }, () => new Array(n).fill(0.12));
    if (kind === "stripes") {
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) I[r][c] = c < 12 ? (c % 2 ? 0.85 : 0.15) : (Math.floor((c - 12) / 3) % 2 ? 0.85 : 0.15);
      return I;
    }
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      if (r >= 4 && r <= 11 && c >= 4 && c <= 11) I[r][c] = 0.85;                 // bright square
      else if ((r - 16) ** 2 + (c - 15.5) ** 2 <= 25) I[r][c] = 0.55;              // gray disk
    }
    if (kind === "noisy") {
      const rng = NB.rng(0x9e3779b9); for (let i = 0; i < 8; i++) rng();
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) I[r][c] = Math.min(1, Math.max(0, I[r][c] + 0.1 * rng.normal()));
    }
    return I;
  }

  // ---- kernels ----
  const sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
  function kernels() {
    const s = st.sigma, g = [];
    for (let i = -2; i <= 2; i++) g.push([-2, -1, 0, 1, 2].map((j) => Math.exp(-(i * i + j * j) / (2 * s * s))));
    const z = g.flat().reduce((a, b) => a + b, 0);
    return {
      box: [[[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9))],
      gauss: [g.map((r) => r.map((v) => v / z))],
      sharpen: [[0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === 1 && j === 1 ? 2 : 0) - 1 / 9))],   // 2·impulse − box blur
      sobelx: [sobelx], sobely: [sobely], gradmag: [sobelx, sobely],
      laplacian: [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]],
    }[st.kernel];
  }
  const signed = () => ["sobelx", "sobely", "laplacian"].includes(st.kernel);
  const at = (I, r, c) => I[Math.min(n - 1, Math.max(0, r))][Math.min(n - 1, Math.max(0, c))];
  const corr = (I, K, r, c) => { const h = (K.length - 1) / 2; let s = 0; for (let i = -h; i <= h; i++) for (let j = -h; j <= h; j++) s += K[i + h][j + h] * at(I, r + i, c + j); return s; };
  function filter(I, Ks) {
    return I.map((row, r) => row.map((_, c) => Ks.length === 2 ? Math.hypot(corr(I, Ks[0], r, c), corr(I, Ks[1], r, c)) : corr(I, Ks[0], r, c)));
  }

  // ---- controls ----
  const ctl = NB.controls(el);
  NB.select(ctl, "kernel", [["box", "box blur 3×3"], ["gauss", "Gaussian blur 5×5"], ["sharpen", "sharpen"], ["sobelx", "Sobel x"],
    ["sobely", "Sobel y"], ["gradmag", "gradient magnitude"], ["laplacian", "Laplacian"]], st.kernel, (v) => { st.kernel = v; draw(); });
  const sig = NB.slider(ctl, { label: "Gaussian σ", min: 0.4, max: 2.5, step: 0.05, value: st.sigma }, (v) => { st.sigma = v; draw(); });
  NB.select(ctl, "image", [["shapes", "shapes"], ["noisy", "shapes + noise"], ["stripes", "stripes"]], st.image, (v) => { st.image = v; draw(); });
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);

  const gray = (v) => { const g = Math.round(255 * Math.min(1, Math.max(0, v))); return `rgb(${g},${g},${g})`; };
  const xIn = 0, xOut = W - IW, xK = IW + 20, KW = xOut - 20 - xK;
  let I, O, Ks, omax;

  function draw() {
    sig.input.disabled = st.kernel !== "gauss";
    I = makeImage(st.image); Ks = kernels(); O = filter(I, Ks);
    omax = Math.max(1e-9, ...O.flat().map(Math.abs));
    NB.clear(svg);
    NB.el("text", { x: xIn + IW / 2, y: 14, "text-anchor": "middle", class: "nb-label" }, svg, "input image");
    NB.el("text", { x: xOut + IW / 2, y: 14, "text-anchor": "middle", class: "nb-label" }, svg, "output");
    NB.el("text", { x: xK + KW / 2, y: 14, "text-anchor": "middle", class: "nb-label" }, svg, "kernel weights");
    // input
    I.forEach((row, r) => row.forEach((v, c) => NB.el("rect", { x: xIn + c * cs, y: top + r * cs, width: cs + 0.4, height: cs + 0.4, fill: gray(v) }, svg)));
    // output: gray for smoothing / magnitude, orange (+) and blue (−) for signed derivatives
    const og = NB.el("g", { style: { cursor: "crosshair" } }, svg);
    NB.el("rect", { x: xOut, y: top, width: IW, height: IW, fill: NB.c.bg }, og);
    O.forEach((row, r) => row.forEach((v, c) => {
      const a = { x: xOut + c * cs, y: top + r * cs, width: cs + 0.4, height: cs + 0.4 };
      if (signed()) Object.assign(a, { fill: v >= 0 ? NB.c.orange : NB.c.blue, "fill-opacity": Math.abs(v) / omax });
      else a.fill = gray(st.kernel === "gradmag" ? v / omax : v);
      const p = NB.el("rect", a, og);
      const pick = () => { st.sel = [r, c]; mark(); };
      p.addEventListener("mouseenter", pick); p.addEventListener("click", pick);
    }));
    [xIn, xOut].forEach((x) => NB.el("rect", { x, y: top, width: IW, height: IW, fill: "none", stroke: NB.c.grid }, svg));
    NB.el("text", { x: xOut + IW / 2, y: top + IW + 17, "text-anchor": "middle", class: "nb-tick" }, svg,
      signed() ? `orange +, blue −; strongest |value| ${NB.fmt(omax, 2)}` : st.kernel === "gradmag" ? `scaled so the largest (${NB.fmt(omax, 2)}) is white` : "values clipped to [0, 1]");
    NB.el("text", { x: xIn + IW / 2, y: top + IW + 17, "text-anchor": "middle", class: "nb-tick" }, svg, "black 0, white 1");
    // kernel grids
    const kg = NB.el("g", {}, svg);
    const grid = (K, y0, size, label) => {
      const m = K.length, c = Math.min(size / m, 34), x0 = xK + (KW - c * m) / 2, wmax = Math.max(...K.flat().map(Math.abs));
      if (label) NB.el("text", { x: xK + KW / 2, y: y0 - 4, "text-anchor": "middle", class: "nb-tick" }, kg, label);
      K.forEach((row, i) => row.forEach((w, j) => {
        NB.el("rect", { x: x0 + j * c, y: y0 + i * c, width: c, height: c, fill: w >= 0 ? NB.c.orange : NB.c.blue, "fill-opacity": 0.08 + 0.4 * Math.abs(w) / wmax, stroke: NB.c.grid }, kg);
        const t = Math.abs(w) < 1e-9 ? "0" : Number.isInteger(w) ? String(w) : w.toFixed(2).replace(/^(-?)0\./, "$1.");
        NB.el("text", { x: x0 + (j + 0.5) * c, y: y0 + (i + 0.5) * c + 4, "text-anchor": "middle", "font-size": 11 }, kg, t.replace("-", "−"));
      }));
      return c * m;
    };
    if (Ks.length === 2) { grid(Ks[0], top + 14, 96, "Gx (Sobel x)"); grid(Ks[1], top + 14 + 96 + 24, 96, "Gy (Sobel y)"); }
    else {
      const h = Ks[0].length * Math.min(KW / Ks[0].length, 34);
      grid(Ks[0], top + (IW - h) / 2, KW);
      const sum = Ks[0].flat().reduce((a, b) => a + b, 0);
      NB.el("text", { x: xK + KW / 2, y: top + (IW + h) / 2 + 18, "text-anchor": "middle", class: "nb-tick" }, kg, `weights sum to ${NB.fmt(Math.abs(sum) < 1e-9 ? 0 : sum, 2)}`);
    }
    hi = NB.el("g", { "pointer-events": "none" }, svg);
    mark();
  }
  let hi;
  const fx = (v) => (Math.abs(v) < 5e-4 ? "0" : v.toFixed(3).replace("-", "−"));
  const sq = (v) => (Math.abs(v) < 1e-9 ? "0²" : v < 0 ? `(−${(-v).toFixed(2)})²` : `${v.toFixed(2)}²`);
  function mark() {
    NB.clear(hi);
    const [r, c] = st.sel, h = (Ks[0].length - 1) / 2;
    NB.el("rect", { x: xIn + (c - h) * cs, y: top + (r - h) * cs, width: (2 * h + 1) * cs, height: (2 * h + 1) * cs, fill: "none", stroke: NB.c.pink, "stroke-width": 2.4 }, hi);
    NB.el("rect", { x: xOut + c * cs, y: top + r * cs, width: cs, height: cs, fill: "none", stroke: NB.c.pink, "stroke-width": 2.4 }, hi);
    const v = O[r][c], where = `row ${r}, column ${c}`;
    const what = st.kernel === "gradmag"
      ? `√(Gx² + Gy²) = √(${sq(corr(I, Ks[0], r, c))} + ${sq(corr(I, Ks[1], r, c))}) = <b>${fx(v)}</b>`
      : `Σ weight × patch = <b>${fx(v)}</b>`;
    const sum = Ks.length === 1 ? Ks[0].flat().reduce((a, b) => a + b, 0) : 0;
    const why = Ks.length === 2 ? "large on edges in any direction, 0 on flat regions."
      : Math.abs(sum - 1) < 1e-9 ? "the weights sum to 1, so flat regions keep their value."
      : "the weights sum to 0, so flat regions give 0 and only changes in intensity respond.";
    read.innerHTML = `output pixel (${where}), input there ${I[r][c].toFixed(2)}: ${what}. ${why[0].toUpperCase() + why.slice(1)}` +
      (st.kernel === "gauss" ? ` A 5×5 kernel is about 6σ wide at σ ≈ 0.8; σ = ${NB.fmt(st.sigma, 2)} ${6 * st.sigma > 5.5 ? "is cut off at the kernel's edge" : "fits"}${st.sigma >= 2 ? ", so the weights get flatter, closer to a box blur" : ""}.` : "");
  }
  draw();
});

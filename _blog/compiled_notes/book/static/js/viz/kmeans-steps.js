/* kmeans-steps: K-means one half-step at a time. E assigns every point to its nearest center, M moves each
 * center to the mean of its points. The distortion J = Σ_n Σ_i z_n^i ||x_n - μ_i||² never goes up.
 * params: {k: 5, seed: 5} */
NB.register("kmeans-steps", function (el, P) {
  const W = 620, H = 380, m = { l: 14, r: 14, t: 10, b: 10 };
  const X = NB.scale(-1, 11, m.l, W - m.r), Y = NB.scale(-0.5, 7, H - m.b, m.t);
  // five separated blobs
  const rng = NB.rng(5), blobs = [[1.8, 1.6, 0.5], [2.2, 5.4, 0.5], [6.2, 1.2, 0.55], [9.0, 5.2, 0.5], [5.8, 4.8, 0.5]];
  const data = [];
  blobs.forEach(([cx, cy, s]) => { for (let i = 0; i < 30; i++) data.push([cx + s * rng.normal(), cy + s * rng.normal()]); });
  const st = { k: P.k || 5, seed: P.seed || 5, init: "random", centers: [], z: null, phase: "E", iter: 0, hist: [] };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "K", min: 2, max: 6, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; restart(); });
  NB.select(ctl, "start", [["random", "random points"], ["pp", "k-means++"]], st.init, (v) => { st.init = v; restart(); });
  const stepBtn = NB.button(ctl, "E: assign", () => { step(); draw(); });
  NB.button(ctl, "run to convergence", () => { for (let i = 0; i < 200 && !converged(); i++) step(); draw(); });
  NB.button(ctl, "new start", () => { st.seed++; restart(); });
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const J = () => (st.z ? data.reduce((s, p, i) => s + d2(p, st.centers[st.z[i]]), 0) : null);
  let lastMoved = true;
  const converged = () => st.z && st.phase === "E" && !lastMoved;

  function restart() {
    const r = NB.rng(1000 + st.seed * 17 + st.k);
    if (st.init === "random") {
      const idx = new Set(); while (idx.size < st.k) idx.add(Math.floor(r() * data.length));
      st.centers = [...idx].map((i) => data[i].slice());
    } else { // k-means++: next center drawn with probability proportional to squared distance to the nearest chosen one
      st.centers = [data[Math.floor(r() * data.length)].slice()];
      while (st.centers.length < st.k) {
        const w = data.map((p) => Math.min(...st.centers.map((c) => d2(p, c)))), tot = w.reduce((a, b) => a + b, 0);
        let u = r() * tot, i = 0; while (u > w[i] && i < data.length - 1) u -= w[i++];
        st.centers.push(data[i].slice());
      }
    }
    st.z = null; st.phase = "E"; st.iter = 0; st.hist = []; lastMoved = true; draw();
  }
  function step() {
    if (st.phase === "E") {
      const z = data.map((p) => { let b = 0; st.centers.forEach((c, i) => { if (d2(p, c) < d2(p, st.centers[b])) b = i; }); return b; });
      lastMoved = !st.z || z.some((v, i) => v !== st.z[i]);
      st.z = z; st.phase = "M"; st.iter++;
    } else {
      st.centers = st.centers.map((c, i) => { const pts = data.filter((_, n) => st.z[n] === i); return pts.length ? [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length] : c; });
      st.phase = "E";
    }
    st.hist.push(J());
  }
  function draw() {
    NB.clear(svg);
    NB.el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "none", stroke: NB.c.grid }, svg);
    data.forEach((p, i) => NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3.6, fill: st.z ? NB.colors[st.z[i]] : NB.c.muted, "fill-opacity": st.z ? 0.75 : 0.5 }, svg));
    st.centers.forEach((c, i) => {
      const g = NB.el("g", { transform: `translate(${X(c[0])},${Y(c[1])})` }, svg);
      NB.el("circle", { r: 9, fill: NB.c.bg, stroke: NB.colors[i], "stroke-width": 3 }, g);
      NB.el("path", { d: "M-5,-5L5,5M-5,5L5,-5", stroke: NB.colors[i], "stroke-width": 2.4 }, g);
    });
    stepBtn.textContent = st.phase === "E" ? "E: assign points" : "M: move centers";
    const j = J();
    const hist = st.hist.filter((v) => v !== null).map((v) => NB.fmt(v, 1));
    read.innerHTML = !st.z ? `K = ${st.k} centers placed. Press <b>E</b> to assign each point to its nearest center.`
      : converged() ? `<b>converged</b> after ${st.iter - 1} iterations: J = <b>${NB.fmt(j, 1)}</b>. Try <b>new start</b>: a different start can end at a different (worse or better) J.`
      : `iteration ${st.iter}, next step <b>${st.phase}</b> · distortion J: ${hist.slice(-6).join(" → ")}`;
  }
  restart();
});

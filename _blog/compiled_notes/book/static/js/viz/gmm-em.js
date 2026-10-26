/* gmm-em: EM for a Gaussian mixture p(x|θ) = Σ_i π_i N(x | μ_i, Σ_i) on 2-d data, one half-step at a time.
 * E computes responsibilities r_n^i = π_i N(x_n|μ_i,Σ_i) / Σ_j π_j N(x_n|μ_j,Σ_j) (drawn as a pie per point);
 * M refits π_i, μ_i, Σ_i as r-weighted averages. The log-likelihood l(θ|x) = Σ_n log Σ_i π_i N(x_n|μ_i,Σ_i)
 * never goes down from one iteration to the next (right panel).
 * params: {k: 3, seed: 2} */
NB.register("gmm-em", function (el, P) {
  const W = 640, H = 380, m = { l: 10, t: 10, b: 10 }, PW = 420;
  const X = NB.scale(0.6, 11.4, m.l, m.l + PW), ppu = PW / 10.8;
  const Y = NB.scale(0, (H - m.t - m.b) / ppu, H - m.b, m.t);   // same units per pixel on both axes
  // three clusters with different shapes, so a mixture with full covariances fits better than round clusters
  const rng = NB.rng(11), data = [];
  const blobs = [ // center, cholesky factor [[a,0],[b,c]], count
    [[3.2, 3.0], [[1.2, 0], [1.0, 0.5]], 70],
    [[5.4, 5.4], [[0.8, 0], [0, 0.8]], 50],
    [[6.8, 2.4], [[1.4, 0], [0, 0.45]], 60]];
  blobs.forEach(([c, L, n]) => { for (let i = 0; i < n; i++) { const a = rng.normal(), b = rng.normal(); data.push([c[0] + L[0][0] * a, c[1] + L[1][0] * a + L[1][1] * b]); } });
  const N = data.length;
  const st = { k: P.k || 3, seed: P.seed || 2, mu: [], S: [], pi: [], r: null, phase: "E", iter: 0, ll: [] };

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "components K", min: 2, max: 4, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; restart(); });
  const stepBtn = NB.button(ctl, "E step", () => { step(); draw(); });
  NB.button(ctl, "run to convergence", () => { for (let i = 0; i < 1000 && !converged(); i++) step(); draw(); });
  NB.button(ctl, "new start", () => { st.seed++; restart(); });
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);

  // ---- math ----
  const pdf = (x, mu, S) => {
    const a = S[0][0], b = S[0][1], c = S[1][1], det = a * c - b * b;
    const dx = x[0] - mu[0], dy = x[1] - mu[1], q = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
    return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
  };
  const loglik = () => data.reduce((s, x) => s + Math.log(st.pi.reduce((t, p, i) => t + p * pdf(x, st.mu[i], st.S[i]), 0) || 1e-300), 0);
  const converged = () => st.ll.length > 1 && st.phase === "E" && st.ll[st.ll.length - 1] - st.ll[st.ll.length - 2] < 1e-4;

  function restart() {
    // hashed seed, a few draws discarded: NB.rng's first draws are correlated for nearby small seeds
    const r = NB.rng(Math.imul(st.seed * 7919 + st.k, 2654435761) >>> 0); for (let i = 0; i < 8; i++) r();
    const idx = new Set(); while (idx.size < st.k) idx.add(Math.floor(r() * N));
    st.mu = [...idx].map((i) => data[i].slice());
    st.S = st.mu.map(() => [[1, 0], [0, 1]]);
    st.pi = st.mu.map(() => 1 / st.k);
    st.r = null; st.phase = "E"; st.iter = 0; st.ll = [loglik()];
    draw();
  }
  function step() {
    if (st.phase === "E") {
      st.r = data.map((x) => { const w = st.pi.map((p, i) => p * pdf(x, st.mu[i], st.S[i])), z = w.reduce((a, b) => a + b, 0) || 1; return w.map((v) => v / z); });
      st.phase = "M";
    } else {
      for (let i = 0; i < st.k; i++) {
        const Ni = st.r.reduce((s, r) => s + r[i], 0) + 1e-10;
        const mu = [0, 1].map((d) => data.reduce((s, x, n) => s + st.r[n][i] * x[d], 0) / Ni);
        const S = [[0, 0], [0, 0]];
        data.forEach((x, n) => { const dx = [x[0] - mu[0], x[1] - mu[1]]; for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) S[a][b] += st.r[n][i] * dx[a] * dx[b] / Ni; });
        S[0][0] += 1e-3; S[1][1] += 1e-3;    // keeps a component from collapsing onto one point
        st.mu[i] = mu; st.S[i] = S; st.pi[i] = Ni / N;
      }
      st.phase = "E"; st.iter++; st.ll.push(loglik());
    }
  }

  // ---- drawing ----
  function ellipse(g, mu, S, k, attrs) {
    const a = S[0][0], b = S[0][1], c = S[1][1], h = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    const l1 = (a + c) / 2 + h, l2 = Math.max(1e-9, (a + c) / 2 - h), th = 0.5 * Math.atan2(2 * b, a - c) * 180 / Math.PI;
    NB.el("ellipse", Object.assign({ cx: X(mu[0]), cy: Y(mu[1]), rx: k * Math.sqrt(l1) * ppu, ry: k * Math.sqrt(l2) * ppu,
      transform: `rotate(${-th} ${X(mu[0])} ${Y(mu[1])})`, fill: "none" }, attrs), g);
  }
  function pie(g, cx, cy, R, w) {
    const top = w.indexOf(Math.max(...w));
    NB.el("circle", { cx, cy, r: R + 0.8, fill: NB.c.bg }, g);
    if (w[top] > 0.98) { NB.el("circle", { cx, cy, r: R, fill: NB.colors[top] }, g); return; }
    let a0 = -Math.PI / 2;
    w.forEach((v, i) => {
      if (v < 0.005) return;
      const a1 = a0 + 2 * Math.PI * v, big = v > 0.5 ? 1 : 0;
      NB.el("path", { d: `M${cx},${cy}L${(cx + R * Math.cos(a0)).toFixed(2)},${(cy + R * Math.sin(a0)).toFixed(2)}A${R},${R} 0 ${big} 1 ${(cx + R * Math.cos(a1)).toFixed(2)},${(cy + R * Math.sin(a1)).toFixed(2)}Z`, fill: NB.colors[i] }, g);
      a0 = a1;
    });
  }
  function draw() {
    NB.clear(svg);
    NB.el("rect", { x: m.l, y: m.t, width: PW, height: H - m.t - m.b, fill: "none", stroke: NB.c.grid }, svg);
    const clip = NB.el("clipPath", { id: "gmm-clip-" + uid }, NB.el("defs", {}, svg));
    NB.el("rect", { x: m.l, y: m.t, width: PW, height: H - m.t - m.b }, clip);
    const g = NB.el("g", { "clip-path": `url(#gmm-clip-${uid})` }, svg);
    data.forEach((x, n) => {
      if (st.r) pie(g, X(x[0]), Y(x[1]), 5.4, st.r[n]);
      else NB.el("circle", { cx: X(x[0]), cy: Y(x[1]), r: 4, fill: NB.c.muted, "fill-opacity": 0.55 }, g);
    });
    for (let i = 0; i < st.k; i++) {
      ellipse(g, st.mu[i], st.S[i], 1, { stroke: NB.colors[i], "stroke-width": 2.4 });
      ellipse(g, st.mu[i], st.S[i], 2, { stroke: NB.colors[i], "stroke-width": 1.4, "stroke-dasharray": "5 4" });
      const c = NB.el("g", { transform: `translate(${X(st.mu[i][0])},${Y(st.mu[i][1])})` }, g);
      NB.el("circle", { r: 7, fill: NB.c.bg, stroke: NB.colors[i], "stroke-width": 2.6 }, c);
      NB.el("text", { y: 4, "text-anchor": "middle", "font-size": 11, "font-weight": 700 }, c, (100 * st.pi[i]).toFixed(0));
    }
    // log-likelihood per iteration
    const lx0 = 500, lx1 = 630, ly0 = H - 44, ly1 = 26;
    const n = st.ll.length, lo = Math.min(...st.ll), hi = Math.max(...st.ll), pad = Math.max(1, (hi - lo) * 0.08);
    const LX = NB.scale(0, Math.max(5, n - 1), lx0, lx1), LY = NB.scale(lo - pad, hi + pad, ly0, ly1);
    NB.axes(svg, LX, LY, { xticks: 3, yticks: 5, xlabel: "iteration", yfmt: (t) => t.toFixed(0) });
    NB.el("text", { x: (lx0 + lx1) / 2, y: 14, "text-anchor": "middle", class: "nb-label" }, svg, "log-likelihood l(θ|x)");
    NB.el("path", { d: NB.line(st.ll.map((v, i) => [i, v]), LX, LY), fill: "none", stroke: NB.c.fg, "stroke-width": 1.8 }, svg);
    st.ll.forEach((v, i) => NB.el("circle", { cx: LX(i), cy: LY(v), r: 2.6, fill: NB.c.fg }, svg));

    stepBtn.textContent = st.phase === "E" ? "E step: responsibilities" : "M step: refit π, μ, Σ";
    const ll = st.ll[n - 1], soft = st.r ? st.r.filter((r) => Math.max(...r) < 0.9).length : 0;
    read.innerHTML = !st.r ? `K = ${st.k} Gaussians started at random points with Σ = I. Log-likelihood l = <b>${ll.toFixed(1)}</b>. Press <b>E</b> to compute each point's responsibilities.`
      : converged() ? `<b>converged</b> after ${st.iter} iterations: l = <b>${ll.toFixed(1)}</b>; ${soft} of ${N} points still have no component above 90% responsibility (where clusters overlap). Try <b>new start</b>: some starts stop at a lower l.`
      : `iteration ${st.iter}, next step <b>${st.phase}</b> · l = <b>${ll.toFixed(1)}</b>` + (n > 1 ? ` (up ${NB.fmt(ll - st.ll[n - 2], 2)} in the last iteration)` : "") +
        ` · ${soft} points split between components (no responsibility above 90%)`;
  }
  const uid = Math.random().toString(36).slice(2, 8);
  restart();
});

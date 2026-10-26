/* fourier-partial-sums: a periodic signal rebuilt from its first K harmonics (a Fourier series),
 * with the power spectrum (squared coefficient magnitudes) below.
 * params: {wave: "square" | "sawtooth" | "triangle", K: 5} */
NB.register("fourier-partial-sums", function (el, P) {
  const PI = Math.PI;
  const waves = {   // f on one period (-π, π) and its sine coefficients b_k: f = Σ b_k sin(kx)
    square: { label: "square", f: (x) => Math.sign(Math.sin(x)), b: (k) => (k % 2 ? 4 / (PI * k) : 0) },
    sawtooth: { label: "sawtooth", f: (x) => { const y = ((x + PI) % (2 * PI) + 2 * PI) % (2 * PI) - PI; return y / PI; }, b: (k) => ((k % 2 ? 2 : -2) / (PI * k)) },
    triangle: { label: "triangle", f: (x) => (2 / PI) * Math.asin(Math.sin(x)), b: (k) => (k % 2 ? ((((k - 1) / 2) % 2 ? -8 : 8) / (PI * PI * k * k)) : 0) },
  };
  const st = { wave: P.wave || "square", K: P.K ?? 5 };
  const ctl = NB.controls(el);
  NB.select(ctl, "signal", Object.entries(waves).map(([k, v]) => [k, v.label]), st.wave, (v) => { st.wave = v; draw(); });
  NB.slider(ctl, { label: "harmonics K", min: 1, max: 60, step: 1, value: st.K, format: (v) => v.toFixed(0) }, (v) => { st.K = v; draw(); });
  const W = 640, H1 = 250, H2 = 150, m = { l: 44, r: 10, t: 10, b: 34 };
  const svg = NB.svg(el, W, H1), svg2 = NB.svg(el, W, H2);
  NB.legend(el, [["signal", NB.c.muted], ["sum of the first K harmonics", NB.c.blue], ["power |b_k|² of harmonic k", NB.c.orange]]);
  const read = NB.readout(el);

  function draw() {
    NB.clear(svg); NB.clear(svg2);
    const Wv = waves[st.wave], xs = NB.range(-2 * PI, 2 * PI, 1200);
    const sum = (x) => { let s = 0; for (let k = 1; k <= st.K; k++) s += Wv.b(k) * Math.sin(k * x); return s; };
    const Xu = NB.scale(-2, 2, m.l, W - m.r), X = (x) => Xu(x / PI), Y = NB.scale(-1.5, 1.5, H1 - m.b, m.t);
    NB.axes(svg, Xu, Y, { xlabel: "x (two periods)", xticks: 4, yticks: 4, xfmt: (t) => (t === 0 ? "0" : t === 1 ? "π" : t === -1 ? "−π" : t + "π") });
    // draw the signal without vertical jump segments
    let d = "", prev = null;
    xs.forEach((x) => { const y = Wv.f(x); d += (prev === null || Math.abs(y - prev) > 0.5 ? "M" : "L") + X(x).toFixed(1) + "," + Y(y).toFixed(1); prev = y; });
    NB.el("path", { d, fill: "none", stroke: NB.c.muted, "stroke-width": 3, "stroke-opacity": 0.6 }, svg);
    NB.el("path", { d: NB.line(xs.map((x) => [x / PI, sum(x)]), Xu, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2 }, svg);
    // power spectrum
    const kmax = 30, X2 = NB.scale(0.5, kmax + 0.5, m.l, W - m.r), Y2 = NB.scale(0, 1.7, H2 - m.b, m.t);
    NB.axes(svg2, X2, Y2, { xlabel: "harmonic k", ylabel: "power", xticks: 6, yticks: 2, grid: false });
    const bw = (X2(1) - X2(0)) * 0.6;
    for (let k = 1; k <= kmax; k++) {
      const p = Wv.b(k) ** 2;
      NB.el("rect", { x: X2(k) - bw / 2, y: Y2(p), width: bw, height: Y2(0) - Y2(p), fill: NB.c.orange, "fill-opacity": k <= st.K ? 0.9 : 0.2 }, svg2);
    }
    // mean squared error over one period
    const one = NB.range(-PI, PI, 4001);
    const mse = one.reduce((s, x) => s + (Wv.f(x) - sum(x)) ** 2, 0) / one.length;
    read.innerHTML = `${st.K} harmonic${st.K > 1 ? "s" : ""} (only k ≤ 30 shown in the spectrum): mean squared error <b>${NB.fmt(mse, 4)}</b>` +
      (st.wave !== "sawtooth" ? " · even k have zero power: this signal has only odd harmonics" : "");
  }
  draw();
});

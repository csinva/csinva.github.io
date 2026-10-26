/* sta-cone-mosaic: recovering a ganglion cell's receptive field with a spike-triggered average.
 * The true receptive field k (24x24 pixels) is a sum of small Gaussian blobs, one per cone the cell connects to,
 * normalized to |k| = 1. Each white-noise frame s has i.i.d. N(0,1) pixels; the cell's drive is u = k.s and its spike
 * count in that frame is Poisson with rate r0 exp(beta u) (an LNP model). The STA is sum_t n_t s_t / sum_t n_t; for
 * Gaussian frames its expectation is beta k, and each pixel's noise shrinks like 1/sqrt(number of spikes).
 * params: {frames: 3} (index into FRAMES) */
NB.register("sta-cone-mosaic", function (el, P) {
  const N = 24, D = N * N, BETA = 1.5, R0 = 0.1;
  const FRAMES = [30, 100, 300, 1000, 3000, 10000, 30000];
  const cones = [[11.6, 11.4, 1.0], [14.8, 11.9, 0.9], [8.4, 11.0, 0.8], [13.3, 8.6, 0.95], [10.0, 8.2, 0.75],
                 [13.0, 14.5, 0.85], [9.7, 14.1, 0.7], [16.4, 9.0, 0.6]];          // [col, row, weight], about one cone spacing apart
  const k = new Float64Array(D);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
    k[i * N + j] = cones.reduce((a, [cx, cy, w]) => a + w * Math.exp(-((j - cx) ** 2 + (i - cy) ** 2) / (2 * 0.55 * 0.55)), 0);
  const kn = Math.sqrt(k.reduce((a, v) => a + v * v, 0));
  for (let i = 0; i < D; i++) k[i] /= kn;

  // white-noise frames generated on demand, accumulated into sum_t n_t s_t
  const rng = NB.rng(0x5a17c0de); for (let i = 0; i < 8; i++) rng();
  const sum = new Float64Array(D), frame = new Float64Array(D);
  let done = 0, spikes = 0, lastFrame = null;
  const snaps = {};                                   // FRAMES[i] -> {sum, spikes, frame}: the slider can go back
  const poisson = (lam) => { const L = Math.exp(-lam); let n = 0, p = rng(); while (p > L) { n++; p *= rng(); } return n; };
  function advance(T) {
    while (done < T) {
      let u = 0;
      for (let i = 0; i < D; i++) { frame[i] = rng.normal(); u += k[i] * frame[i]; }
      const n = poisson(R0 * Math.exp(BETA * u));
      if (n) { spikes += n; for (let i = 0; i < D; i++) sum[i] += n * frame[i]; lastFrame = Float64Array.from(frame); }
      done++;
      if (FRAMES.includes(done)) snaps[done] = { sum: Float64Array.from(sum), spikes, frame: lastFrame };
    }
    return snaps[T];
  }

  const st = { fi: P.frames ?? 3 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "white-noise frames shown", min: 0, max: FRAMES.length - 1, step: 1, value: st.fi, format: (v) => FRAMES[v].toLocaleString("en-US") }, (v) => { st.fi = v; draw(); });

  const W = 660, H = 250, svg = NB.svg(el, W, H), PX = 180, GAP = 36, X0 = 15, Y0 = 34;
  NB.legend(el, [["brighter than average (+)", NB.c.orange], ["darker (−)", NB.c.blue]]);
  const read = NB.readout(el);
  const titles = ["a frame that drew a spike", "spike-triggered average", "true receptive field"];
  const panels = titles.map((t, p) => {
    const x0 = X0 + p * (PX + GAP), c = PX / N, cells = [];
    NB.el("text", { x: x0 + PX / 2, y: Y0 - 10, class: "nb-tick", "text-anchor": "middle" }, svg, t);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
      cells.push(NB.el("rect", { x: (x0 + j * c).toFixed(2), y: (Y0 + i * c).toFixed(2), width: (c + 0.3).toFixed(2), height: (c + 0.3).toFixed(2), stroke: "none" }, svg));
    NB.el("rect", { x: x0, y: Y0, width: PX, height: PX, fill: "none", stroke: NB.c.grid }, svg);
    return cells;
  });
  const paint = (cells, vals, scale) => cells.forEach((c, i) => {
    const v = vals[i] / scale;
    c.setAttribute("fill", v >= 0 ? NB.c.orange : NB.c.blue);
    c.setAttribute("fill-opacity", Math.min(1, Math.abs(v)).toFixed(3));
  });
  const kmax = Math.max(...k);
  NB.el("text", { x: X0 + 2 * (PX + GAP) + PX / 2, y: Y0 + PX + 20, class: "nb-tick", "text-anchor": "middle" }, svg, "one blob per cone");
  const note = NB.el("text", { x: X0 + PX + GAP + PX / 2, y: Y0 + PX + 20, class: "nb-tick", "text-anchor": "middle" }, svg, "");

  function draw() {
    const { sum, spikes, frame: shown } = advance(FRAMES[st.fi]);
    const sta = Array.from(sum, (v) => (spikes ? v / spikes : 0));
    const stam = Math.sqrt(sta.reduce((a, v) => a + v * v, 0));
    const corr = stam ? sta.reduce((a, v, i) => a + v * k[i], 0) / stam : 0;      // cosine similarity with k (|k| = 1)
    paint(panels[0], shown || new Float64Array(D), 2.5);
    paint(panels[1], sta, Math.max(1e-9, Math.max(...sta.map(Math.abs))));
    paint(panels[2], k, kmax);
    note.textContent = `average of ${spikes.toLocaleString("en-US")} spike-triggering frames`;
    read.innerHTML = `${FRAMES[st.fi].toLocaleString("en-US")} frames, <b>${spikes.toLocaleString("en-US")}</b> spikes; ` +
      `similarity of the STA to the true field (cosine) <b>${corr.toFixed(2)}</b>; ` +
      `noise per pixel ≈ 1/√spikes = <b>${spikes ? (1 / Math.sqrt(spikes)).toFixed(3) : "–"}</b> vs. a peak signal of ${(BETA * kmax).toFixed(2)}`;
  }
  draw();
});

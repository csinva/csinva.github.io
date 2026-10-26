/* grad-flow: backpropagate one gradient through a randomly initialized deep net (width 64, weights N(0, g²/64))
 * and plot the gradient's size at every layer. Sigmoids shrink it at every layer (σ' ≤ 1/4); with a large g
 * it grows instead. params: {act: "sigmoid", gain: 1, depth: 20} */
NB.register("grad-flow", function (el, P) {
  const w = 64, depth = P.depth || 20, reps = 6;
  const acts = {
    sigmoid: { f: (a) => 1 / (1 + Math.exp(-a)), df: (a) => { const s = 1 / (1 + Math.exp(-a)); return s * (1 - s); } },
    tanh: { f: Math.tanh, df: (a) => 1 - Math.tanh(a) ** 2 },
    relu: { f: (a) => (a > 0 ? a : 0), df: (a) => (a > 0 ? 1 : 0) },
  };
  const st = { act: P.act || "sigmoid", g: P.gain ?? 1 };
  const ctl = NB.controls(el);
  NB.select(ctl, "activation", [["sigmoid", "sigmoid"], ["tanh", "tanh"], ["relu", "ReLU"]], st.act, (v) => { st.act = v; draw(); });
  NB.slider(ctl, { label: "weight scale g", min: 0.5, max: 3, step: 0.05, value: st.g }, (v) => { st.g = v; draw(); });
  const W = 640, H = 320, m = { l: 66, r: 12, t: 10, b: 36 };
  const svg = NB.svg(el, W, H), read = NB.readout(el);
  const X = NB.scale(1, depth, m.l, W - m.r), Y = NB.scale(-20, 8, H - m.b, m.t);
  // fixed standard-normal draws, rescaled by g, so the slider changes the scale and not the sample
  const rng = NB.rng(9);
  const Z = Array.from({ length: depth }, () => Array.from({ length: w }, () => Array.from({ length: w }, () => rng.normal() / Math.sqrt(w))));
  const inputs = Array.from({ length: reps }, () => Array.from({ length: w }, () => rng.normal()));
  const outGrad = Array.from({ length: reps }, () => Array.from({ length: w }, () => rng.normal()));
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

  function draw() {
    NB.clear(svg);
    const A = acts[st.act], g = st.g, logs = Array.from({ length: depth }, () => 0);
    for (let r = 0; r < reps; r++) {
      let h = inputs[r]; const pre = [];
      for (let l = 0; l < depth; l++) { const a = Z[l].map((row) => g * row.reduce((s, z, j) => s + z * h[j], 0)); pre.push(a); h = a.map(A.f); }
      let dh = outGrad[r].slice(); const n0 = norm(dh);
      for (let l = depth - 1; l >= 0; l--) {
        logs[l] += Math.log10(Math.max(norm(dh) / n0, 1e-300)) / reps;   // gradient arriving at layer l's output
        const da = dh.map((d, i) => d * A.df(pre[l][i]));
        dh = Array.from({ length: w }, (_, j) => g * Z[l].reduce((s, row, i) => s + row[j] * da[i], 0));
      }
    }
    NB.axes(svg, X, Y, { xlabel: "layer (output on the right)", xticks: 6, yticks: 6, yfmt: (t) => (t === 0 ? "1" : "1e" + t) });
    // own y title: the tick labels ("1e-20") are wide, so it sits further left than NB.axes would put it
    NB.el("text", { x: 0, y: 0, class: "nb-label", "text-anchor": "middle", transform: `translate(14,${(m.t + H - m.b) / 2}) rotate(-90)` }, svg, "gradient size vs. output");
    NB.el("line", { x1: X(1), x2: X(depth), y1: Y(0), y2: Y(0), stroke: NB.c.muted, "stroke-dasharray": "4 3" }, svg);
    const pts = logs.map((v, l) => [l + 1, Math.max(-20, Math.min(8, v))]);
    NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: NB.c.pink, "stroke-width": 2.4 }, svg);
    pts.forEach((p) => NB.el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.6, fill: NB.c.pink }, svg));
    const first = logs[0], perLayer = Math.pow(10, first / (depth - 1));
    const verdict = first < -2 ? "<b>vanishing</b>: the first layers barely learn" : first > 2 ? "<b>exploding</b>" : "roughly <b>stable</b>";
    read.innerHTML = `at layer 1 the gradient is <b>${first < -19.9 ? "< 1e-20" : NB.fmt(Math.pow(10, first), 2)}</b> × its size at the output (about ×${NB.fmt(perLayer, 2)} per layer): ${verdict}`;
  }
  draw();
});

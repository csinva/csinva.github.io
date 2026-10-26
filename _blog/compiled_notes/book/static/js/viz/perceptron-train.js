/* perceptron-train: the perceptron on 2-d data with labels y = ±1 and a bias feature, x = (1, x1, x2).
 * Cycle through the points in a fixed order; whenever y_i θᵀx_i ≤ 0 (a mistake), update θ ← θ + y_i x_i.
 * Left: the data, the boundary θᵀx = 0 after k updates, θ's direction, and the point the next update uses.
 * Right: number of misclassified training points after each update. With separable data it reaches 0 and
 * the algorithm stops; with two flipped labels it never does.
 * params: {k: 3, data: "separable"} */
NB.register("perceptron-train", function (el, P) {
  const n = 24, maxK = 40;
  function makeData(sep) {
    // points at least 0.25 from the line 0.8 x1 + x2 = 1.2 (illustrative), labeled by its side
    const r = NB.rng(29), X = [], Y = [], h = Math.hypot(0.8, 1);
    while (X.length < n) {
      const x1 = -3 + 6 * r(), x2 = -3 + 6 * r(), s = (0.8 * x1 + x2 - 1.2) / h;
      if (Math.abs(s) < 0.25) continue;
      X.push([1, x1, x2]); Y.push(s > 0 ? 1 : -1);
    }
    if (!sep) [3, 11].forEach((i) => (Y[i] = -Y[i]));   // two flipped labels: no line separates the data
    return { X, Y };
  }
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function train({ X, Y }) {
    // steps[k] = state after k updates: θ, mistakes, and the point the next update will use (or -1 if done)
    let th = [0, 0, 0], i = 0, clean = 0;
    const mistakes = (t) => X.filter((x, j) => Y[j] * dot(t, x) <= 0).length;
    const nextMistake = () => { for (let c = 0; c < n; c++) { const j = (i + c) % n; if (Y[j] * dot(th, X[j]) <= 0) return j; } return -1; };
    const steps = [{ th: th.slice(), m: mistakes(th), next: nextMistake() }];
    while (steps.length <= maxK) {
      const j = nextMistake();
      if (j < 0) break;
      th = th.map((t, d) => t + Y[j] * X[j][d]);
      i = (j + 1) % n;
      steps.push({ th: th.slice(), m: mistakes(th), used: j, next: nextMistake() });
    }
    return steps;
  }
  const sets = { separable: makeData(true), noisy: makeData(false) };
  const runs = { separable: train(sets.separable), noisy: train(sets.noisy) };
  const st = { k: P.k ?? 3, data: P.data || "separable" };

  const ctl = NB.controls(el);
  NB.select(ctl, "data", [["separable", "linearly separable"], ["noisy", "not separable (2 flipped labels)"]], st.data, (v) => { st.data = v; st.k = Math.min(st.k, runs[v].length - 1); kS.set(st.k); draw(); });
  const kS = NB.slider(ctl, { label: "updates k", min: 0, max: maxK, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; draw(); });
  NB.button(ctl, "next update", () => { st.k = Math.min(maxK, st.k + 1); kS.set(st.k); draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px" } }, el);
  const S = 360, s1 = NB.svg(row, S, S), W2 = 340, H2 = 360, s2 = NB.svg(row, W2, H2);
  NB.legend(el, [["y = +1", NB.c.blue], ["y = −1", NB.c.purple], ["boundary θᵀx = 0", NB.c.fg], ["next point used", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(-3.2, 3.2, 8, S - 8), Y = NB.scale(-3.2, 3.2, S - 8, 8);
  const m = { l: 44, r: 10, t: 12, b: 40 };
  const Xk = NB.scale(0, maxK, m.l, W2 - m.r), Ym = NB.scale(0, n, H2 - m.b, m.t);

  function mark(g, x, y, lab, r, extra) {
    const a = Object.assign({ fill: lab > 0 ? NB.c.blue : NB.c.purple, "fill-opacity": 0.85 }, extra || {});
    if (lab > 0) NB.el("circle", Object.assign({ cx: x, cy: y, r }, a), g);
    else NB.el("rect", Object.assign({ x: x - r * 0.88, y: y - r * 0.88, width: r * 1.76, height: r * 1.76 }, a), g);
  }

  function draw() {
    const steps = runs[st.data], D = sets[st.data], K = Math.min(st.k, steps.length - 1), s = steps[K], th = s.th;
    const done = s.next < 0;
    NB.clear(s1); NB.clear(s2);
    NB.el("rect", { x: 8, y: 8, width: S - 16, height: S - 16, fill: "none", stroke: NB.c.grid }, s1);
    const nrm = Math.hypot(th[1], th[2]);
    if (nrm > 1e-9) {
      // shade the side predicted +1 (θᵀx > 0): clip the half-plane to the box
      const box = [[-3.2, -3.2], [3.2, -3.2], [3.2, 3.2], [-3.2, 3.2]], poly = [];
      for (let a = 0; a < 4; a++) {
        const p = box[a], q = box[(a + 1) % 4], fp = dot(th, [1, ...p]), fq = dot(th, [1, ...q]);
        if (fp > 0) poly.push(p);
        if ((fp > 0) !== (fq > 0)) { const t = fp / (fp - fq); poly.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
      }
      if (poly.length > 2) NB.el("polygon", { points: poly.map((p) => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" "), fill: NB.c.blue, "fill-opacity": 0.08 }, s1);
      const ends = [];
      for (let a = 0; a < 4; a++) {
        const p = box[a], q = box[(a + 1) % 4], fp = dot(th, [1, ...p]), fq = dot(th, [1, ...q]);
        if ((fp > 0) !== (fq > 0)) { const t = fp / (fp - fq); ends.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
      }
      if (ends.length === 2) {
        NB.el("line", { x1: X(ends[0][0]), y1: Y(ends[0][1]), x2: X(ends[1][0]), y2: Y(ends[1][1]), stroke: NB.c.fg, "stroke-width": 2.2 }, s1);
        // θ's direction (θ1, θ2), drawn from the middle of the visible boundary toward the +1 side
        const c = [(ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2], L = 0.8;
        const tip = [c[0] + (L * th[1]) / nrm, c[1] + (L * th[2]) / nrm];
        NB.el("line", { x1: X(c[0]), y1: Y(c[1]), x2: X(tip[0]), y2: Y(tip[1]), stroke: NB.c.fg, "stroke-width": 1.8 }, s1);
        const ang = Math.atan2(-(Y(tip[1]) - Y(c[1])), X(tip[0]) - X(c[0]));
        const hx = X(tip[0]), hy = Y(tip[1]), hl = 8;
        NB.el("polygon", { points: [[hx, hy], [hx - hl * Math.cos(ang - 0.45), hy + hl * Math.sin(ang - 0.45)], [hx - hl * Math.cos(ang + 0.45), hy + hl * Math.sin(ang + 0.45)]].map((p) => p.join(",")).join(" "), fill: NB.c.fg }, s1);
        NB.el("text", { x: hx + 6 * Math.cos(ang), y: hy - 6 * Math.sin(ang) + 5, "font-size": 15, "font-style": "italic", "text-anchor": Math.cos(ang) >= 0 ? "start" : "end" }, s1, "θ");
      }
    }
    D.X.forEach((x, j) => {
      const wrong = D.Y[j] * dot(th, x) <= 0;
      mark(s1, X(x[1]), Y(x[2]), D.Y[j], 5.5, wrong ? { "fill-opacity": 0.35, stroke: lab(D.Y[j]), "stroke-width": 1.6 } : {});
    });
    if (!done) {
      const x = D.X[s.next];
      NB.el("circle", { cx: X(x[1]), cy: Y(x[2]), r: 11, fill: "none", stroke: NB.c.orange, "stroke-width": 2.6 }, s1);
    }

    // mistakes vs. updates
    NB.axes(s2, Xk, Ym, { xlabel: "update k", ylabel: "misclassified points", xticks: 4, yticks: 4 });
    const pts = steps.map((t, k) => [k, t.m]);
    NB.el("path", { d: NB.line(pts, Xk, Ym), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, s2);
    pts.forEach((p) => NB.el("circle", { cx: Xk(p[0]), cy: Ym(p[1]), r: 2.4, fill: NB.c.orange }, s2));
    NB.el("line", { x1: Xk(K), x2: Xk(K), y1: Ym(0), y2: Ym(n), stroke: NB.c.muted, "stroke-dasharray": "3 3" }, s2);
    if (steps.length - 1 < maxK) NB.el("text", { x: Xk(steps.length - 1) + 6, y: Ym(0) - 8, "font-size": 12.5, fill: NB.c.muted }, s2, "converged: no mistakes left");

    const f = (v) => NB.fmt(v, 2), thS = `θ = (${th.map(f).join(", ")})`;
    let txt = `after <b>${K}</b> updates, ${thS}: <b>${s.m}</b> of ${n} points misclassified. `;
    if (st.k > K) txt = `converged after <b>${K}</b> updates, ${thS}: every point is on the right side, so there are no more updates. `;
    else if (done) txt += "Every point is on the right side, so the perceptron stops here.";
    else {
      const x = D.X[s.next], y = D.Y[s.next];
      txt += `Next update uses the circled point (y = ${y > 0 ? "+1" : "−1"}, y θᵀx = ${f(y * dot(th, x))} ≤ 0): θ ← θ ${y > 0 ? "+" : "−"} (1, ${f(x[1])}, ${f(x[2])}).`;
    }
    read.innerHTML = txt;
  }
  const lab = (y) => (y > 0 ? NB.c.blue : NB.c.purple);
  draw();
});

/* knapsack-table: the note's bottom-up 0/1 knapsack, filling the table K[i][w] one cell at a time.
 * K[i][w] = best value using the first i items with capacity w:
 *   K[i][w] = max(val[i-1] + K[i-1][w - wt[i-1]], K[i-1][w])   if wt[i-1] <= w, else K[i-1][w].
 * The current cell is outlined; the two cells it reads are marked ("skip" item i: the cell above, "take" item i:
 * the cell one row up and wt[i-1] columns left). When the table is full, the traceback marks the chosen items.
 * Items: weights [1, 3, 4, 5], values [1, 4, 5, 7]. K[n][W] checked against a brute force over all 16 subsets
 * for every W = 0..8 (e.g. W = 7 -> 9 with items 2 and 3, W = 6 -> 8 with items 1 and 4, W = 8 -> 11 with items 2 and 4).
 * params: {W: 7} */
NB.register("knapsack-table", function (el, P) {
  const wt = [1, 3, 4, 5], val = [1, 4, 5, 7], n = wt.length;
  const st = { W: P.W ?? 7, t: 0 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "capacity W", min: 1, max: 8, step: 1, value: st.W, format: (v) => v.toFixed(0) }, (v) => { st.W = v; rerun(); });
  const ctl2 = NB.controls(el);
  NB.button(ctl2, "◀ back", () => { stop(); st.t = Math.max(0, st.t - 1); draw(); });
  NB.button(ctl2, "step ▶", () => { stop(); st.t = Math.min(frames.length - 1, st.t + 1); draw(); });
  const play = NB.button(ctl2, "▶ play", () => toggle());
  NB.button(ctl2, "fill all", () => { stop(); st.t = frames.length - 1; draw(); });
  const svg = NB.svg(el, 640, 276);
  NB.legend(el, [["current cell", NB.c.orange], ["skip item i: K[i−1][w]", NB.c.blue], ["take item i: val + K[i−1][w−wt]", NB.c.green], ["traceback", NB.c.pink]]);
  const read = NB.readout(el);
  let frames = [], K = [], timer = null;

  function rerun() {
    stop();
    const W = st.W;
    K = Array.from({ length: n + 1 }, () => Array(W + 1).fill(0));
    frames = [];
    for (let i = 0; i <= n; i++)
      for (let w = 0; w <= W; w++) {
        const f = { i, w, skip: null, take: null, won: null };
        if (i === 0 || w === 0) K[i][w] = 0;
        else if (wt[i - 1] <= w) {
          const a = val[i - 1] + K[i - 1][w - wt[i - 1]], b = K[i - 1][w];
          f.skip = [i - 1, w]; f.take = [i - 1, w - wt[i - 1]]; f.a = a; f.b = b;
          K[i][w] = Math.max(a, b); f.won = a > b ? "take" : "skip";
        } else { K[i][w] = K[i - 1][w]; f.skip = [i - 1, w]; f.won = "skip"; f.tooBig = true; }
        frames.push(f);
      }
    // traceback: walk up from K[n][W]; the value changes between rows exactly when item i was taken
    const path = [], chosen = [];
    let w = W;
    for (let i = n; i > 0; i--) {
      path.push([i, w]);
      if (K[i][w] !== K[i - 1][w]) { chosen.push(i); w -= wt[i - 1]; }
    }
    path.push([0, w]);
    frames.push({ done: true, path, chosen: chosen.reverse() });
    st.t = 0; draw();
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; } }
  function toggle() {
    if (timer) return stop();
    if (st.t >= frames.length - 1) st.t = 0;
    play.textContent = "❚❚ pause";
    timer = setInterval(() => { st.t = Math.min(frames.length - 1, st.t + 1); draw(); if (st.t >= frames.length - 1) stop(); }, 450);
  }

  function draw() {
    NB.clear(svg);
    const W = st.W, f = frames[st.t];
    const x0 = 168, y0 = 46, cw = Math.min(54, (632 - x0) / (W + 1)), ch = 44;
    const filled = (i, w) => f.done || i * (W + 1) + w <= st.t;
    // column headers (capacity w) and row headers (items)
    NB.el("text", { x: x0 + ((W + 1) * cw) / 2, y: 14, class: "nb-label", "text-anchor": "middle" }, svg, "capacity w");
    for (let w = 0; w <= W; w++) NB.el("text", { x: x0 + w * cw + cw / 2, y: y0 - 10, "font-size": 14, fill: NB.c.muted, "text-anchor": "middle" }, svg, w);
    NB.el("text", { x: 4, y: y0 - 10, "font-size": 14, fill: NB.c.muted }, svg, "item i");
    for (let i = 0; i <= n; i++) {
      const lab = i === 0 ? "0: no items" : `${i}: wt ${wt[i - 1]}, val ${val[i - 1]}`;
      const chosen = f.done && f.chosen.includes(i);
      NB.el("text", { x: 4, y: y0 + i * ch + ch / 2 + 5, "font-size": 16, fill: chosen ? NB.c.pink : undefined, "font-weight": chosen ? 700 : 400 }, svg, lab);
    }
    const onPath = (i, w) => f.done && f.path.some(([a, b]) => a === i && b === w);
    for (let i = 0; i <= n; i++)
      for (let w = 0; w <= W; w++) {
        const x = x0 + w * cw, y = y0 + i * ch;
        let stroke = NB.c.grid, sw = 1, fill = "transparent", fo = 0;
        if (!f.done) {
          if (f.skip && f.skip[0] === i && f.skip[1] === w) { fill = NB.c.blue; fo = f.won === "skip" ? 0.35 : 0.14; }
          if (f.take && f.take[0] === i && f.take[1] === w) { fill = NB.c.green; fo = f.won === "take" ? 0.35 : 0.14; }
        } else if (onPath(i, w)) { fill = NB.c.pink; fo = 0.25; }
        NB.el("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 4, fill, "fill-opacity": fo, stroke, "stroke-width": sw }, svg);
        if (!f.done && f.i === i && f.w === w) NB.el("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 4, fill: "none", stroke: NB.c.orange, "stroke-width": 3 }, svg);
        if (filled(i, w)) NB.el("text", { x: x + cw / 2, y: y + ch / 2 + 6, "text-anchor": "middle", "font-size": 19, "font-weight": (!f.done && f.i === i && f.w === w) ? 700 : 400 }, svg, K[i][w]);
      }
    if (f.done) {
      const tw = f.chosen.reduce((s, i) => s + wt[i - 1], 0);
      read.innerHTML = `Table full: <b>K[${n}][${W}] = ${K[n][W]}</b>. Walking up from the corner, the value changes between rows i−1 and i exactly when item i was taken: ` +
        `items <b>${f.chosen.join(", ") || "none"}</b> (total weight ${tw} ≤ ${W}).`;
    } else if (f.i === 0 || f.w === 0) {
      read.innerHTML = `K[${f.i}][${f.w}] = 0: base case (${f.i === 0 ? "no items" : "no capacity"}).`;
    } else if (f.tooBig) {
      read.innerHTML = `K[${f.i}][${f.w}]: item ${f.i} weighs ${wt[f.i - 1]} &gt; ${f.w}, so it cannot be taken; copy the cell above: <b>${K[f.i][f.w]}</b>.`;
    } else {
      read.innerHTML = `K[${f.i}][${f.w}] = max(take: ${val[f.i - 1]} + K[${f.i - 1}][${f.w - wt[f.i - 1]}] = ${f.a}, skip: K[${f.i - 1}][${f.w}] = ${f.b}) = <b>${K[f.i][f.w]}</b>` +
        ` (${f.a === f.b ? "tie" : f.won === "take" ? "take item " + f.i : "skip item " + f.i}).`;
    }
  }
  rerun();
});

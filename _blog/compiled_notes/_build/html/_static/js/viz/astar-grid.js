/* astar-grid: best-first search on a grid with 8-way moves (diagonals cost √2) and the octile-distance
 * heuristic h, which is admissible and consistent. The select switches the evaluation function:
 * uniform-cost f = g, A* f = g + h, greedy best-first f = h. Click a cell to add or remove a wall.
 * The slider replays the expansion order. Path cost is compared with the optimal (uniform-cost) cost.
 * params: {mode: "astar" | "ucs" | "greedy"} */
NB.register("astar-grid", function (el, P) {
  const W = 28, H = 15, R2 = Math.SQRT2, S = [2, 7], G = [25, 7];
  const key = (x, y) => y * W + x;
  const walls = new Set();
  for (let y = 3; y < 12; y++) walls.add(key(18, y));
  for (let x = 12; x < 19; x++) { walls.add(key(x, 3)); walls.add(key(x, 11)); }
  for (let y = 0; y < 5; y++) walls.add(key(7, y));
  for (let y = 10; y < 15; y++) walls.add(key(7, y));
  const init = [...walls];
  const st = { mode: P.mode || "astar", step: Infinity };
  const modes = [["ucs", "uniform-cost: f = g"], ["astar", "A*: f = g + h"], ["greedy", "greedy best-first: f = h"]];
  const ctl = NB.controls(el);
  NB.select(ctl, "search", modes, st.mode, (v) => { st.mode = v; st.step = Infinity; draw(); });
  const stepSl = NB.slider(ctl, { label: "expansions shown", min: 1, max: 400, step: 1, value: 400, format: (v) => v.toFixed(0) }, (v) => { st.step = v; draw(false); });
  NB.button(ctl, "reset walls", () => { walls.clear(); init.forEach((k) => walls.add(k)); st.step = Infinity; draw(); });
  const cs = 22, m = 4, svg = NB.svg(el, W * cs + 2 * m, H * cs + 2 * m);
  NB.legend(el, [["expanded (closed set)", NB.c.blue], ["frontier", NB.c.orange], ["path found", NB.c.green]]);
  const read = NB.readout(el);

  const oct = (x, y) => { const dx = Math.abs(x - G[0]), dy = Math.abs(y - G[1]); return Math.max(dx, dy) + (R2 - 1) * Math.min(dx, dy); };
  function search(mode) {
    const f = (g, x, y) => (mode === "ucs" ? g : mode === "greedy" ? oct(x, y) : g + oct(x, y));
    const s = key(...S), goal = key(...G);
    const g = new Map([[s, 0]]), fv = new Map([[s, f(0, ...S)]]), par = new Map(), closed = [], isClosed = new Set();
    const seenAt = new Map([[s, 0]]); // when each cell first entered the frontier (in number of expansions)
    const heap = [[fv.get(s), oct(...S), 0, s]];
    let c = 0;
    while (heap.length) {
      let bi = 0; // small grid: a linear scan for the minimum is fast enough
      for (let i = 1; i < heap.length; i++) {
        const a = heap[i], b = heap[bi];
        if (a[0] < b[0] - 1e-12 || (Math.abs(a[0] - b[0]) <= 1e-12 && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])))) bi = i;
      }
      const [ff, , , p] = heap.splice(bi, 1)[0];
      if (isClosed.has(p) || ff > fv.get(p) + 1e-12) continue;
      isClosed.add(p); closed.push(p);
      if (p === goal) break;
      const px = p % W, py = Math.floor(p / W);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        if (!dx && !dy) continue;
        const qx = px + dx, qy = py + dy, q = key(qx, qy);
        if (qx < 0 || qx >= W || qy < 0 || qy >= H || walls.has(q) || isClosed.has(q)) continue;
        if (dx && dy && (walls.has(key(px + dx, py)) || walls.has(key(px, py + dy)))) continue; // no corner cutting
        const ng = g.get(p) + (dx && dy ? R2 : 1), nf = f(ng, qx, qy);
        if (!fv.has(q) || nf < fv.get(q) - 1e-12) {
          g.set(q, ng); fv.set(q, nf); par.set(q, p); c++; heap.push([nf, oct(qx, qy), c, q]);
          if (!seenAt.has(q)) seenAt.set(q, closed.length);
        }
      }
    }
    const found = isClosed.has(goal);
    return { closed, seenAt, par, cost: found ? g.get(goal) : null, found };
  }

  let res;
  function draw(recompute = true) {
    if (recompute) {
      res = search(st.mode);
      stepSl.input.max = res.closed.length;
      st.step = Math.min(st.step, res.closed.length);
      stepSl.set(st.step);
    }
    NB.clear(svg);
    const n = Math.min(st.step, res.closed.length), done = n === res.closed.length;
    const closedNow = new Set(res.closed.slice(0, n));
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const k = key(x, y);
      let fill = "transparent", op = 1;
      if (walls.has(k)) { fill = NB.c.fg; op = 0.7; }
      else if (closedNow.has(k)) { fill = NB.c.blue; op = 0.35; }
      else if (res.seenAt.has(k) && res.seenAt.get(k) < n) { fill = NB.c.orange; op = 0.45; }
      const r = NB.el("rect", { x: m + x * cs + 0.5, y: m + y * cs + 0.5, width: cs - 1, height: cs - 1, fill, "fill-opacity": op, stroke: NB.c.grid, "stroke-width": 0.6, style: { cursor: "pointer" } }, svg);
      r.addEventListener("click", () => {
        if ((x === S[0] && y === S[1]) || (x === G[0] && y === G[1])) return;
        walls.has(k) ? walls.delete(k) : walls.add(k); st.step = Infinity; draw();
      });
    }
    if (done && res.found) {
      const pts = []; let p = key(...G);
      while (p !== undefined) { pts.push([p % W, Math.floor(p / W)]); p = res.par.get(p); }
      NB.el("path", { d: pts.map((q, i) => (i ? "L" : "M") + (m + (q[0] + 0.5) * cs) + "," + (m + (q[1] + 0.5) * cs)).join(""), fill: "none", stroke: NB.c.green, "stroke-width": 3.5, "stroke-linejoin": "round", "pointer-events": "none" }, svg);
    }
    [[S, "S"], [G, "G"]].forEach(([q, t]) => {
      NB.el("circle", { cx: m + (q[0] + 0.5) * cs, cy: m + (q[1] + 0.5) * cs, r: cs * 0.42, fill: NB.c.fg, "pointer-events": "none" }, svg);
      NB.el("text", { x: m + (q[0] + 0.5) * cs, y: m + (q[1] + 0.5) * cs + 4.5, "text-anchor": "middle", style: { fill: NB.c.bg }, "font-size": 13, "font-weight": 700, "pointer-events": "none" }, svg, t);
    });
    if (!done) { read.innerHTML = `${n} of ${res.closed.length} expansions shown. Move the slider to the right end to see the path.`; return; }
    if (!res.found) { read.innerHTML = `<b>no path</b>: the walls cut G off from S. Click a wall to remove it.`; return; }
    const best = st.mode === "ucs" ? res.cost : search("ucs").cost;
    const extra = res.cost - best;
    read.innerHTML = `expanded <b>${res.closed.length}</b> nodes · path cost <b>${NB.fmt(res.cost, 2)}</b> ` +
      (extra < 1e-9 ? `(optimal)` : `(optimal is ${NB.fmt(best, 2)}, so this path costs <b>${NB.fmt(extra, 2)}</b> more)`) + `. Click cells to add or remove walls.`;
  }
  draw();
});

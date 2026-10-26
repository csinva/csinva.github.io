/* alpha-beta: minimax with alpha-beta pruning on a depth-3 ternary game tree (MAX at the root, then MIN, then MAX,
 * 27 leaves with values 1..20 from a seeded rng). The search is recorded as a list of steps; the step slider
 * replays it. Each interior node shows its current [alpha, beta] window and, once finished, its value; pruned
 * subtrees are greyed out. The ordering preset reorders every node's children by their minimax value: best
 * first (MAX children high to low, MIN children low to high) evaluates the fewest leaves, b^ceil(d/2) +
 * b^floor(d/2) - 1 = 9 + 3 - 1 = 11; worst first prunes nothing (27). Leaf counts checked against a python
 * alpha-beta on the same trees (port of NB.rng).
 * params: {seed: 2, order: "random" | "best" | "worst"} */
NB.register("alpha-beta", function (el, P) {
  const st = { seed: P.seed ?? 2, order: P.order || "random", step: Infinity };
  const B = 3, D = 3;
  function leaves(seed) { const u = NB.rng(seed * 7919 + 1); u(); return Array.from({ length: B ** D }, () => 1 + Math.floor(u() * 20)); }
  // tree node: {kids, val (leaves), isMax, depth, x, y, key}
  function build(vals) {
    let k = 0;
    const mk = (depth) => depth === D ? { leaf: true, val: vals[k++], depth } : { depth, isMax: depth % 2 === 0, kids: Array.from({ length: B }, () => mk(depth + 1)) };
    return mk(0);
  }
  const mm = (n) => n.leaf ? n.val : (n.isMax ? Math.max : Math.min)(...n.kids.map(mm));
  function reorder(n, mode) {
    if (n.leaf || mode === "random") return n;
    n.kids.forEach((c) => reorder(c, mode));
    const sgn = (n.isMax ? -1 : 1) * (mode === "best" ? 1 : -1);
    n.kids.sort((a, b) => sgn * (mm(a) - mm(b)));
    return n;
  }
  const W = 640, H = 318, ys = [34, 118, 202, 282];
  function layout(n) {
    let i = 0;
    (function walk(n, key) {
      n.key = key; n.y = ys[n.depth];
      if (n.leaf) { n.x = 22 + (i++) * ((W - 44) / (B ** D - 1)); return; }
      n.kids.forEach((c, j) => walk(c, key + j));
      n.x = (n.kids[0].x + n.kids[B - 1].x) / 2;
    })(n, "r");
  }
  const fmtv = (v) => (v === Infinity ? "+∞" : v === -Infinity ? "−∞" : String(v));
  // run alpha-beta and record the steps
  function search(root) {
    const ev = [];
    (function ab(n, a, b) {
      if (n.leaf) { ev.push({ key: n.key, kind: "leaf", v: n.val, msg: `evaluate a leaf: <b>${n.val}</b>` }); return n.val; }
      const who = n.isMax ? "MAX" : "MIN";
      ev.push({ key: n.key, kind: "enter", a, b, msg: `enter a ${who} node with window [α, β] = [${fmtv(a)}, ${fmtv(b)}]` });
      let v = n.isMax ? -Infinity : Infinity;
      for (let i = 0; i < B; i++) {
        const cv = ab(n.kids[i], a, b);
        v = n.isMax ? Math.max(v, cv) : Math.min(v, cv);
        const cut = n.isMax ? v >= b : v <= a;
        if (cut) {
          const rest = n.kids.slice(i + 1).map((c) => c.key);
          ev.push({ key: n.key, kind: "cut", a, b, v, pruned: rest, msg: n.isMax
            ? `${who} node now has v = ${v} ≥ β = ${fmtv(b)}: the MIN player above already has a move worth ${fmtv(b)}, so it never comes here. <b>Prune ${rest.length} remaining branch${rest.length === 1 ? "" : "es"}</b>.`
            : `${who} node now has v = ${v} ≤ α = ${fmtv(a)}: the MAX player above already has a move worth ${fmtv(a)}, so it never comes here. <b>Prune ${rest.length} remaining branch${rest.length === 1 ? "" : "es"}</b>.` });
          return v;
        }
        if (n.isMax) a = Math.max(a, v); else b = Math.min(b, v);
        const last = i === B - 1;
        ev.push({ key: n.key, kind: last ? "done" : "update", a, b, v, msg: last
          ? `${who} node finished: value <b>${v}</b>`
          : `${who} node: child returned ${cv}, so v = ${v} and the window is [${fmtv(a)}, ${fmtv(b)}]` });
      }
      return v;
    })(root, -Infinity, Infinity);
    return ev;
  }

  const ctl = NB.controls(el);
  NB.select(ctl, "move ordering", [["random", "as generated"], ["best", "best moves first"], ["worst", "worst moves first"]], st.order, (v) => { st.order = v; st.step = Infinity; draw(); });
  const sl = NB.slider(ctl, { label: "search step", min: 0, max: 80, step: 1, value: 80, format: (v) => v.toFixed(0) }, (v) => { st.step = v; draw(); });
  NB.button(ctl, "next step", () => { st.step = Math.min(st.nev, st.step + 1); draw(); });
  NB.button(ctl, "new leaves", () => { st.seed++; st.step = Infinity; draw(); });
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["MAX node (△)", NB.c.blue], ["MIN node (▽)", NB.c.pink], ["current step", NB.c.orange], ["pruned, never looked at", NB.c.muted, true]]);
  const read = NB.readout(el);

  function draw() {
    NB.clear(svg);
    const root = reorder(build(leaves(st.seed)), st.order);
    layout(root);
    const ev = search(root);
    st.nev = ev.length;
    sl.input.max = ev.length; if (st.step > ev.length) st.step = ev.length; sl.set(st.step);
    // replay the first `step` events
    const S = {}, pruned = new Set();
    ev.slice(0, st.step).forEach((e) => {
      const s = (S[e.key] = S[e.key] || {});
      if (e.kind === "leaf") { s.done = true; s.v = e.v; }
      if (e.a !== undefined) { s.a = e.a; s.b = e.b; }
      if (e.kind === "enter") s.on = true;
      if (e.kind === "update") s.v = e.v;
      if (e.kind === "done" || e.kind === "cut") { s.v = e.v; s.done = true; }
      if (e.kind === "cut") e.pruned.forEach((k) => pruned.add(k));
    });
    const cur = st.step > 0 ? ev[st.step - 1] : null;
    const isPruned = (key) => [...pruned].some((p) => key.startsWith(p));
    const nodes = [];
    (function walk(n) { nodes.push(n); if (!n.leaf) n.kids.forEach(walk); })(root);
    // edges
    nodes.forEach((n) => { if (!n.leaf) n.kids.forEach((c) => {
      const pr = isPruned(c.key), seen = S[c.key];
      NB.el("line", { x1: n.x, y1: n.y + 13, x2: c.x, y2: c.y - (c.leaf ? 11 : 13), stroke: pr ? NB.c.muted : NB.c.fg,
        "stroke-opacity": pr ? 0.45 : seen ? 0.75 : 0.3, "stroke-width": seen && !pr ? 1.6 : 1, "stroke-dasharray": pr ? "3 4" : null }, svg);
    }); });
    // nodes
    nodes.forEach((n) => {
      const s = S[n.key] || {}, pr = isPruned(n.key), isCur = cur && cur.key === n.key;
      const col = n.leaf ? NB.c.fg : n.isMax ? NB.c.blue : NB.c.pink;
      const g = NB.el("g", { opacity: pr ? 0.35 : 1 }, svg);
      const stroke = isCur ? NB.c.orange : pr ? NB.c.muted : col, sw = isCur ? 3 : 1.6;
      const fill = pr ? "none" : s.done || s.on ? col : "none", fo = s.done ? 0.22 : 0.08;
      if (n.leaf) {
        NB.el("rect", { x: n.x - 10, y: n.y - 11, width: 20, height: 22, rx: 3, fill: s.done ? col : "none", "fill-opacity": 0.12, stroke, "stroke-width": sw, "stroke-dasharray": pr ? "3 3" : null }, g);
        NB.el("text", { x: n.x, y: n.y + 4.5, "text-anchor": "middle", "font-size": 12.5, "font-weight": s.done ? 700 : 400, style: { fill: s.done ? NB.c.fg : NB.c.muted } }, g, n.val);
        return;
      }
      const r = 15, pts = n.isMax ? `${n.x},${n.y - r} ${n.x + r * 1.1},${n.y + r * 0.8} ${n.x - r * 1.1},${n.y + r * 0.8}`
                                 : `${n.x},${n.y + r} ${n.x + r * 1.1},${n.y - r * 0.8} ${n.x - r * 1.1},${n.y - r * 0.8}`;
      NB.el("polygon", { points: pts, fill, "fill-opacity": fo, stroke, "stroke-width": sw, "stroke-dasharray": pr ? "3 3" : null }, g);
      if (s.v !== undefined && isFinite(s.v)) NB.el("text", { x: n.x, y: n.y + (n.isMax ? 8 : 1), "text-anchor": "middle", "font-size": 12, "font-weight": s.done ? 700 : 400 }, g, s.v);
      if (s.a !== undefined && !pr) {
        const ty = n.depth === 0 ? n.y + 4 : n.y - 20, tx = n.depth === 0 ? n.x + 26 : n.x;
        NB.el("text", { x: tx, y: ty, "text-anchor": n.depth === 0 ? "start" : "middle", "font-size": 11, "paint-order": "stroke", stroke: "var(--nb-bg)", "stroke-width": 4,
          style: { fill: isCur ? NB.c.orange : NB.c.muted } }, g, `[${fmtv(s.a)}, ${fmtv(s.b)}]`);
      }
    });
    // the move chosen at the root, once the search is over
    if (st.step >= ev.length) {
      const best = root.kids.find((c) => mm(c) === mm(root));
      NB.el("line", { x1: root.x, y1: root.y + 13, x2: best.x, y2: best.y - 13, stroke: NB.c.green, "stroke-width": 3.5 }, svg);
    }
    const nleaf = (e) => e.filter((x) => x.kind === "leaf").length;
    const done = nleaf(ev.slice(0, st.step)), total = nleaf(ev);
    const nb = nleaf(search(reorder(build(leaves(st.seed)), "best"))), nw = nleaf(search(reorder(build(leaves(st.seed)), "worst")));
    const head = st.step === 0 ? "Step 0: nothing searched yet. Press next step." : `Step ${st.step}: ${cur.msg}.`.replace(/\.\.$/, ".");
    read.innerHTML = `${head}<br>Leaves evaluated: <b>${done}</b> so far, <b>${total} of ${B ** D}</b> for the whole search ` +
      `(these leaves with best ordering: ${nb}, with worst: ${nw}). Minimax value of the root: <b>${mm(root)}</b>` + (st.step >= ev.length ? " (green: the move MAX picks)." : ".");
  }
  draw();
});

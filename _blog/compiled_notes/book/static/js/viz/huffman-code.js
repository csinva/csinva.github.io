/* huffman-code: build a Huffman tree from the letter counts of a short text. Every merge takes the two lightest
 * trees out of the min-heap and puts back one tree whose weight is their sum; left edge = 0, right edge = 1.
 * Ties go to the tree made first (leaves before merged trees). Shows the forest after k merges (each tree drawn
 * where it sits in the final tree), the heap of current roots, the final prefix codes, and the average code
 * length L = sum p_i l_i against the entropy H = -sum p_i log2 p_i and a fixed-length code of ceil(log2 n) bits.
 * params: {text: "...", merges: null (= all)} */
NB.register("huffman-code", function (el, P) {
  const st = { text: P.text || "she sells sea shells by the sea shore", k: P.merges ?? null };
  const ctl = NB.controls(el);
  const box = NB.el("label", { class: "nb-select", style: { flex: "1 1 18rem" } }, ctl);
  NB.el("span", { class: "nb-slider-label" }, box, "text");
  const input = NB.el("input", { type: "text", value: st.text, maxlength: 120, style: { flex: "1", minWidth: "10rem", background: "var(--nb-panel)", color: "var(--nb-fg)", border: "1px solid var(--nb-border)", borderRadius: "6px", padding: "0.2rem 0.4rem", font: "inherit" } }, box);
  const ctl2 = NB.controls(el);
  const slider = NB.slider(ctl2, { label: "merges done", min: 0, max: 1, step: 1, value: 0, format: (v) => v.toFixed(0) }, (v) => { st.k = v; draw(); });
  NB.button(ctl2, "next merge", () => { st.k = Math.min(st.k + 1, model.merges.length); draw(); });
  const heapDiv = NB.el("div", { style: { margin: "0.2rem 0 0.1rem", fontSize: "0.84rem", color: "var(--nb-muted)", lineHeight: "1.9" } }, el);
  const W = 640, svg = NB.svg(el, W, 280);
  const codeDiv = NB.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.25rem 0.45rem", margin: "0.3rem 0", fontSize: "0.84rem" } }, el);
  const svg2 = NB.svg(el, W, 96);
  const read = NB.readout(el);
  const show = (c) => (c === " " ? "␣" : c);
  const mono = "var(--pst-font-family-monospace)";
  let model = null;

  function build(text) {
    const counts = new Map();
    [...text.toLowerCase()].forEach((c) => { if (c !== "\n") counts.set(c, (counts.get(c) || 0) + 1); });
    const leaves = [...counts.entries()].sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1))
      .map(([sym, w], id) => ({ sym, w, id, syms: show(sym) }));
    let heap = leaves.slice(), id = leaves.length;
    const merges = [];
    const pri = (a, b) => a.w - b.w || a.id - b.id;
    while (heap.length > 1) {
      heap.sort(pri);
      const [a, b] = heap.splice(0, 2), node = { w: a.w + b.w, id: id++, left: a, right: b, syms: a.syms + b.syms };
      merges.push(node); heap.push(node);
    }
    const root = heap[0] || null, codes = {};
    // final layout: leaves in left-to-right order, parents centered over children, y = depth
    let nx = 0, depth = 0;
    (function walk(nd, code, dep) {
      if (!nd) return;
      nd.depth = dep; depth = Math.max(depth, dep);
      if (nd.sym !== undefined) { nd.x = nx++; codes[nd.sym] = code || "0"; return; }
      walk(nd.left, code + "0", dep + 1); walk(nd.right, code + "1", dep + 1);
      nd.x = (nd.left.x + nd.right.x) / 2;
    })(root, "", 0);
    const N = text.length ? [...counts.values()].reduce((a, b) => a + b, 0) : 0;
    const H = leaves.reduce((s, l) => s - (l.w / N) * Math.log2(l.w / N), 0);
    const L = leaves.reduce((s, l) => s + (l.w / N) * codes[l.sym].length, 0);
    const fixed = Math.max(1, Math.ceil(Math.log2(leaves.length)));
    return { leaves, merges, root, codes, depth, N, H, L, fixed, n: leaves.length };
  }

  function draw() {
    NB.clear(svg); NB.clear(svg2); NB.clear(heapDiv); NB.clear(codeDiv);
    const m = model, nm = m.merges.length;
    if (!m.n) { read.innerHTML = "Type some text."; return; }
    const k = Math.min(st.k ?? nm, nm); st.k = k;
    slider.input.max = Math.max(1, nm); slider.set(k);
    const made = new Set(m.merges.slice(0, k));
    const exists = (nd) => nd.sym !== undefined || made.has(nd);
    const last = k ? m.merges[k - 1] : null;
    // current roots = existing nodes whose parent doesn't exist yet
    const roots = m.leaves.concat(m.merges.slice(0, k)).filter((nd) => !m.merges.slice(0, k).some((p) => p.left === nd || p.right === nd))
      .sort((a, b) => a.w - b.w || a.id - b.id);
    const label = (nd) => (nd.syms.length <= 4 ? nd.syms : nd.syms.slice(0, 3) + "…") + " " + nd.w;   // a tree is named by its letters

    NB.el("span", {}, heapDiv, roots.length > 1 ? "min-heap of trees (lightest first): " : "one tree left: ");
    roots.forEach((nd, i) => NB.el("span", { style: { display: "inline-block", fontFamily: mono, fontSize: "0.82rem", padding: "0 0.35rem", margin: "0 2px", borderRadius: "4px",
      border: `1px ${i < 2 && roots.length > 1 ? "solid var(--nb-c-orange)" : "solid var(--nb-border)"}`, background: "var(--nb-panel)", color: i < 2 && roots.length > 1 ? "var(--nb-c-orange)" : "var(--nb-fg)" } }, heapDiv, label(nd)));
    if (roots.length > 1) NB.el("span", {}, heapDiv, " (orange: merged next)");

    // tree
    const top = 20, bot = 238, rowH = Math.min(56, (bot - top) / Math.max(1, m.depth));
    const sx = (x) => (m.n === 1 ? W / 2 : 24 + (x * (W - 48)) / (m.n - 1)), sy = (dp) => top + dp * rowH;
    const r = Math.min(12, Math.max(8, (W - 48) / Math.max(1, m.n - 1) / 2.4));
    svg.setAttribute("viewBox", `0 0 ${W} ${bot + 34}`);
    const nodes = [], edges = [];
    (function walk(nd) { if (!nd) return; if (nd.left) { walk(nd.left); walk(nd.right); } if (exists(nd)) nodes.push(nd); })(m.root);
    nodes.forEach((nd) => { if (nd.left) edges.push([nd, nd.left, "0"], [nd, nd.right, "1"]); });
    edges.forEach(([p, c, b]) => {
      const hot = p === last;
      NB.el("line", { x1: sx(p.x), y1: sy(p.depth), x2: sx(c.x), y2: sy(c.depth), stroke: hot ? NB.c.orange : NB.c.muted, "stroke-width": hot ? 2.2 : 1.2 }, svg);
      const mx = (sx(p.x) + sx(c.x)) / 2, my = (sy(p.depth) + sy(c.depth)) / 2;
      NB.el("text", { x: mx + (b === "0" ? -7 : 7), y: my - 2, "text-anchor": "middle", "font-size": 11, fill: NB.c.muted }, svg, b);
    });
    nodes.forEach((nd) => {
      const leaf = nd.sym !== undefined, hot = nd === last, cx = sx(nd.x), cy = sy(nd.depth);
      NB.el("circle", { cx, cy, r, fill: leaf ? "var(--nb-panel)" : NB.c.bg, stroke: hot ? NB.c.orange : leaf ? NB.c.blue : NB.c.muted, "stroke-width": hot ? 2.4 : 1.4 }, svg);
      NB.el("text", { x: cx, y: cy + 4, "text-anchor": "middle", "font-size": leaf ? 12.5 : 11, "font-weight": leaf ? 600 : 400, "font-family": leaf ? mono : null }, svg, leaf ? show(nd.sym) : nd.w);
      if (leaf) NB.el("text", { x: cx, y: cy + r + 12, "text-anchor": "middle", "font-size": 11, fill: NB.c.muted }, svg, nd.w);
    });

    // codes (only once the tree is finished)
    const done = k === nm;
    [...m.leaves].sort((a, b) => b.w - a.w || a.id - b.id).forEach((l) => {
      const c = NB.el("span", { style: { fontFamily: mono, border: "1px solid var(--nb-border)", borderRadius: "4px", padding: "0 0.35rem", background: "var(--nb-panel)" } }, codeDiv);
      NB.el("b", {}, c, show(l.sym));
      NB.el("span", { style: { color: "var(--nb-muted)" } }, c, ` ×${l.w} `);
      NB.el("span", { style: { color: done ? "var(--nb-c-blue)" : "var(--nb-faint)" } }, c, done ? m.codes[l.sym] : "?");
    });

    // bits per symbol: entropy bound, Huffman, fixed length
    const bars = [["entropy H", m.H, NB.c.purple], ["Huffman L", m.L, NB.c.blue], [`fixed (⌈log₂ ${m.n}⌉)`, m.fixed, NB.c.muted]];
    const X = NB.scale(0, Math.max(m.fixed, m.L, 1) * 1.08, 150, W - 60);
    bars.forEach(([name, v, col], i) => {
      const y = 8 + i * 28, vis = i !== 1 || done;
      NB.el("text", { x: 140, y: y + 14, "text-anchor": "end", "font-size": 12.5 }, svg2, name);
      if (vis) NB.el("rect", { x: X(0), y, width: Math.max(1, X(v) - X(0)), height: 18, rx: 3, fill: col, "fill-opacity": 0.75 }, svg2);
      NB.el("text", { x: vis ? X(v) + 6 : X(0) + 6, y: y + 14, "font-size": 12.5, fill: vis ? null : NB.c.muted }, svg2, vis ? NB.fmt(v, 3) + " bits" : "finish the tree");
    });

    if (!done) {
      const nx = m.merges[k];
      read.innerHTML = `Merge <b>${k + 1}</b> of ${nm} (next): take the two lightest trees, <b>${label(nx.left)}</b> and <b>${label(nx.right)}</b>, ` +
        `and put back one tree of weight <b>${nx.w}</b>. ${roots.length} trees in the heap now.`;
    } else if (m.n === 1) {
      read.innerHTML = `Only one distinct symbol: H = 0, but any code still needs 1 bit per symbol.`;
    } else {
      read.innerHTML = `${m.N} symbols, ${m.n} distinct. Huffman: <b>${NB.fmt(m.L, 3)}</b> bits/symbol on average (${Math.round(m.L * m.N)} bits in all), ` +
        `entropy bound H = <b>${NB.fmt(m.H, 3)}</b>, fixed length <b>${m.fixed}</b> bits (${m.fixed * m.N} in all). H ≤ L < H + 1.`;
    }
  }
  function rebuild() { const wasDone = !model || st.k === null || st.k >= model.merges.length; model = build(st.text); if (wasDone) st.k = model.merges.length; draw(); }
  input.addEventListener("input", () => { st.text = input.value; rebuild(); });
  rebuild();
});

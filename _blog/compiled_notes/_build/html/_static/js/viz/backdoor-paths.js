/* backdoor-paths: the back-door criterion checked on a small DAG. Clicking a measured node toggles it in the
 * adjustment set S. Every path between T and Y in the skeleton is listed and marked blocked/open with the note's
 * d-separation rule: S blocks a path if the path has an arrow-emitting node (chain or fork) in S, or a collider that
 * is not in S and has no descendant in S. S satisfies the back-door criterion if no node of S is a descendant of T
 * and every back-door path (one that starts with an arrow into T) is blocked.
 * The graph: confounder Z (T <- Z -> Y), mediator M (T -> M -> Y), and the M-bias structure from the note
 * (U1 -> T, U1 -> X <- U2, U2 -> Y) with U1, U2 unmeasured.
 * params: {S: []} */
NB.register("backdoor-paths", function (el, P) {
  const nodes = { T: [110, 185], Y: [530, 185], Z: [320, 150], M: [320, 262], X: [320, 56], U1: [150, 56], U2: [490, 56] };
  const edges = [["Z", "T"], ["Z", "Y"], ["T", "M"], ["M", "Y"], ["U1", "T"], ["U1", "X"], ["U2", "X"], ["U2", "Y"]];
  const unmeasured = new Set(["U1", "U2"]), fixed = new Set(["T", "Y"]);
  const S = new Set((P.S || []).filter((v) => !unmeasured.has(v) && !fixed.has(v)));
  const label = (v) => (v[0] === "U" ? "U" + v.slice(1).replace(/\d/, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]) : v);
  const children = (v) => edges.filter((e) => e[0] === v).map((e) => e[1]);
  const descendants = (v) => { const out = new Set(), st = [v]; while (st.length) children(st.pop()).forEach((c) => { if (!out.has(c)) { out.add(c); st.push(c); } }); return out; };
  const dir = (a, b) => edges.some((e) => e[0] === a && e[1] === b);          // is there an edge a -> b
  // all simple paths from T to Y in the skeleton
  const nbrs = (v) => edges.filter((e) => e.includes(v)).map((e) => (e[0] === v ? e[1] : e[0]));
  const paths = [];
  (function dfs(path) {
    const v = path[path.length - 1];
    if (v === "Y") { paths.push(path.slice()); return; }
    nbrs(v).forEach((w) => { if (!path.includes(w)) { path.push(w); dfs(path); path.pop(); } });
  })(["T"]);

  const W = 640, H = 312, R = 23;
  const svg = NB.svg(el, W, H);
  const list = NB.el("div", { class: "nb-readout", style: { marginTop: "0.3rem" } }, el);
  const read = NB.readout(el);

  function status(path) {
    for (let i = 1; i < path.length - 1; i++) {
      const v = path[i], collider = dir(path[i - 1], v) && dir(path[i + 1], v);
      if (!collider && S.has(v)) return { open: false, why: `${label(v)} is in S` };
      if (collider && !S.has(v) && ![...descendants(v)].some((d) => S.has(d))) return { open: false, why: `collider ${label(v)} is not conditioned on` };
    }
    const col = path.slice(1, -1).filter((v, i) => dir(path[i], v) && dir(path[i + 2], v));
    return { open: true, why: col.length ? `conditioning on collider ${col.map(label).join(", ")} opens it` : "nothing on it is in S" };
  }
  const text = (path) => path.map((v, i) => (i ? (dir(path[i - 1], v) ? " → " : " ← ") : "") + label(v)).join("");

  function draw() {
    NB.clear(svg);
    const defs = NB.el("defs", {}, svg);
    const mk = (id, c) => { const k = NB.el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 10, markerHeight: 10, markerUnits: "userSpaceOnUse", orient: "auto" }, defs); NB.el("path", { d: "M0,0L10,5L0,10Z", fill: c }, k); };
    const uid = Math.random().toString(36).slice(2, 7);
    mk("bd-a-" + uid, NB.c.muted); mk("bd-p-" + uid, NB.c.pink); mk("bd-g-" + uid, NB.c.green);
    const info = paths.map((p) => ({ p, back: dir(p[1], "T"), ...status(p) }));
    const hot = new Map();                                     // edge -> color for open paths
    info.forEach((r) => { if (!r.open) return; for (let i = 0; i + 1 < r.p.length; i++) hot.set([r.p[i], r.p[i + 1]].sort().join("|"), r.back ? "p" : "g"); });
    edges.forEach(([a, b]) => {
      const [x1, y1] = nodes[a], [x2, y2] = nodes[b], d = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / d, uy = (y2 - y1) / d;
      const h = hot.get([a, b].sort().join("|")), c = h === "p" ? NB.c.pink : h === "g" ? NB.c.green : NB.c.muted;
      NB.el("line", { x1: x1 + ux * R, y1: y1 + uy * R, x2: x2 - ux * (R + 2), y2: y2 - uy * (R + 2), stroke: c, "stroke-width": h ? 3 : 1.5, "marker-end": `url(#bd-${h || "a"}-${uid})` }, svg);
    });
    const desc = descendants("T");
    Object.entries(nodes).forEach(([v, [x, y]]) => {
      const g = NB.el("g", { style: { cursor: unmeasured.has(v) || fixed.has(v) ? "default" : "pointer" } }, svg);
      const inS = S.has(v);
      NB.el("circle", { cx: x, cy: y, r: R, fill: inS ? NB.c.fg : NB.c.bg, "fill-opacity": inS ? 0.28 : 1, stroke: fixed.has(v) ? NB.c.blue : NB.c.fg, "stroke-width": fixed.has(v) ? 2.4 : 1.4, "stroke-dasharray": unmeasured.has(v) ? "4 3" : null }, g);
      NB.el("text", { x, y: y + 5, "text-anchor": "middle", style: { fontWeight: 600, fontSize: "15px" } }, g, label(v));
      const sub = unmeasured.has(v) ? "unmeasured" : v === "T" ? "treatment" : v === "Y" ? "outcome" : desc.has(v) ? "descendant of T" : "";
      if (sub) NB.el("text", { x, y: unmeasured.has(v) ? y - R - 7 : y + R + 15, "text-anchor": "middle", class: "nb-tick" }, g, sub);
      if (!unmeasured.has(v) && !fixed.has(v)) g.addEventListener("click", () => { if (S.has(v)) S.delete(v); else S.add(v); draw(); });
    });
    list.innerHTML = info.map((r) => `${r.back ? "back-door path" : "causal path"} <b>${text(r.p)}</b>: ` +
      `<span style="color:${r.open ? (r.back ? "var(--nb-c-pink)" : "var(--nb-c-green)") : "var(--nb-muted)"};font-weight:600">${r.open ? "open" : "blocked"}</span> (${r.why})`).join("<br>");
    const bad = [...S].filter((v) => desc.has(v)), openBack = info.filter((r) => r.back && r.open);
    const sTxt = S.size ? `{${[...S].map(label).join(", ")}}` : "{ } (empty)";
    let verdict;
    if (bad.length) verdict = `fails the back-door criterion: ${bad.map(label).join(", ")} is a descendant of T`;
    else if (openBack.length) verdict = `fails the back-door criterion: ${openBack.length} back-door path${openBack.length > 1 ? "s are" : " is"} open`;
    else verdict = `satisfies the back-door criterion, so P(y | do(t)) = Σ<sub>s</sub> P(y | t, s) P(s)`;
    read.innerHTML = `S = <b>${sTxt}</b> ${verdict}.`;
  }
  draw();
});

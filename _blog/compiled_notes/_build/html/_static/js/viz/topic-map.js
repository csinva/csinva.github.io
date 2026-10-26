/* topic-map: every chapter as a node, linked to the chapters it shares the most vocabulary with
 * (tf-idf over the note text, computed by build_book.py). Hover to highlight neighbors, click to open,
 * click a part chip to dim everything else.  params: {data: "_static/data/topic_map.json"} */
NB.register("topic-map", function (el, params) {
  const W = 980, H = 680;
  const partColor = { cs: NB.c.purple, math: NB.c.blue, stat: NB.c.green, ml: NB.c.orange, ai: NB.c.pink, neuro: NB.c.yellow, research: "var(--nb-muted)" };
  const root = document.documentElement.dataset.content_root || document.querySelector("html").getAttribute("data-content_root") || "./";
  const chips = NB.el("div", { class: "nb-chips" }, el);
  const svg = NB.svg(el, W, H, "topic-map");
  const tip = NB.el("div", { class: "nb-tip" }, el);
  fetch(root + params.data).then((r) => r.json()).then((data) => {
    if (!window.d3) { el.textContent = "(topic map needs d3)"; return; }
    const d3 = window.d3;
    let nodes = data.nodes.map((n) => Object.assign({}, n));
    let links = data.links.map((l) => Object.assign({}, l));
    const focusPart = params.focus;           // part pages: only that part plus its closest chapters elsewhere
    if (focusPart) {
      const inPart = new Set(nodes.filter((n) => n.part === focusPart).map((n) => n.id));
      links = links.filter((l) => inPart.has(l.source) || inPart.has(l.target));
      const keep = new Set([...inPart, ...links.map((l) => l.source), ...links.map((l) => l.target)]);
      nodes = nodes.filter((n) => keep.has(n.id));
      nodes.forEach((n) => { n.ext = !inPart.has(n.id); });
      chips.style.display = "none";
    }
    const deg = {}; links.forEach((l) => { deg[l.source] = (deg[l.source] || 0) + 1; deg[l.target] = (deg[l.target] || 0) + 1; });
    const nbrs = {}; links.forEach((l) => { (nbrs[l.source] = nbrs[l.source] || new Set()).add(l.target); (nbrs[l.target] = nbrs[l.target] || new Set()).add(l.source); });

    // pull each part toward its own region so the map reads as clusters
    const parts = data.parts.filter((p) => nodes.some((n) => n.part === p.id));
    const centers = {}; parts.forEach((p, i) => { const a = (2 * Math.PI * i) / parts.length - Math.PI / 2; centers[p.id] = [W / 2 + 290 * Math.cos(a), H / 2 + 190 * Math.sin(a)]; });
    if (focusPart) parts.forEach((p) => { centers[p.id] = p.id === focusPart ? [W / 2, H / 2] : [W / 2, H / 2]; });

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(70).strength((l) => 0.15 + 0.8 * l.w))
      .force("charge", d3.forceManyBody().strength(-260))
      .force("x", d3.forceX((d) => centers[d.part][0]).strength(focusPart ? (d) => (d.ext ? 0.02 : 0.08) : 0.18))
      .force("y", d3.forceY((d) => centers[d.part][1]).strength(focusPart ? (d) => (d.ext ? 0.02 : 0.08) : 0.18))
      .force("collide", d3.forceCollide((d) => 14 + 2.4 * d.title.length).strength(0.9))
      .stop();
    // keep nodes inside the frame on every step, so collisions resolve along the edges too
    for (let i = 0; i < 500; i++) { sim.tick(); nodes.forEach((n) => { n.x = Math.max(60, Math.min(W - 60, n.x)); n.y = Math.max(28, Math.min(H - 16, n.y)); }); }

    // fit the view to the laid-out graph (part maps use a fraction of the frame)
    const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y), pad = 55;
    const bx0 = Math.min(...xs) - pad, by0 = Math.min(...ys) - pad, bw = Math.max(...xs) - Math.min(...xs) + 2 * pad;
    const bh = Math.max(Math.max(...ys) - Math.min(...ys) + 2 * pad, bw * 0.45);
    svg.setAttribute("viewBox", `${bx0} ${by0} ${bw} ${bh}`);
    const gl = NB.el("g", { class: "tm-links" }, svg);
    const ln = links.map((l) => NB.el("line", { x1: l.source.x, y1: l.source.y, x2: l.target.x, y2: l.target.y,
      class: "tm-link" + (l.source.ext || l.target.ext ? " ext" : ""), "stroke-width": 0.6 + 4 * l.w }, gl));
    const gn = NB.el("g", { class: "tm-nodes" }, svg);
    const nd = nodes.map((n) => {
      const a = NB.el("a", { href: root + n.url, class: "tm-node" + (n.ext ? " ext" : "") }, gn);
      NB.el("circle", { cx: n.x, cy: n.y, r: 5 + 1.6 * Math.sqrt(deg[n.id] || 1), style: { fill: partColor[n.part] } }, a);
      NB.el("text", { x: n.x, y: n.y - 10, "text-anchor": "middle", class: "tm-label" }, a, n.title);
      a.addEventListener("mouseenter", () => focus(n));
      a.addEventListener("mouseleave", () => focus(null));
      return { n, a };
    });
    let partFilter = null;
    function focus(n) {
      const keep = n ? new Set([n.id, ...(nbrs[n.id] || [])]) : null;
      nd.forEach(({ n: m, a }) => a.classList.toggle("dim", (keep && !keep.has(m.id)) || (!keep && partFilter && m.part !== partFilter)));
      ln.forEach((line, i) => {
        const l = links[i], on = n && (l.source.id === n.id || l.target.id === n.id);
        line.classList.toggle("hot", !!on);
        line.classList.toggle("dim", (n && !on) || (!n && partFilter && (l.source.part !== partFilter || l.target.part !== partFilter)));
      });
      tip.textContent = n ? `${n.title}  ·  closest: ${[...(nbrs[n.id] || [])].map((id) => nodes.find((m) => m.id === id).title).join(", ")}` : "";
      tip.classList.toggle("on", !!n);
    }
    parts.forEach((p) => {
      const c = NB.el("button", { type: "button", class: "nb-chip" }, chips);
      NB.el("span", { class: "nb-dot", style: { background: partColor[p.id] } }, c);
      NB.el("span", {}, c, p.title);
      c.addEventListener("click", () => {
        partFilter = partFilter === p.id ? null : p.id;
        chips.querySelectorAll(".nb-chip").forEach((b) => b.classList.toggle("active", b === c && partFilter));
        focus(null);
      });
    });
  });
});

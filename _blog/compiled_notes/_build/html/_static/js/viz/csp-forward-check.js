/* csp-forward-check: backtracking search for a 3-coloring of a small constraint graph (12 variables,
 * binary "different color" constraints). Three solvers: plain backtracking (fixed variable order, a value is
 * legal if no assigned neighbor has it), forward checking (after each assignment, delete that value from the
 * unassigned neighbors' domains and back up at once if one becomes empty), and forward checking with the
 * minimum-remaining-values heuristic (pick the unassigned variable with the fewest values left, ties by degree
 * on unassigned neighbors, then by number). Values are always tried in the order blue, orange, green.
 * A step slider replays the search one assignment / backtrack at a time.
 * params: {method: "bt" | "fc" | "mrv", step: -1 (= last)} */
NB.register("csp-forward-check", function (el, P) {
  const K = 3, N = 12;
  const pos = Array.from({ length: N }, (_, i) => [i % 4, Math.floor(i / 4)]);
  const E = [[0, 4], [0, 5], [1, 2], [2, 3], [2, 5], [2, 6], [3, 7], [4, 5], [4, 9], [5, 6], [5, 9], [6, 9], [7, 10], [8, 9], [10, 11]];
  const nb = Array.from({ length: N }, () => []);
  E.forEach(([a, b]) => { nb[a].push(b); nb[b].push(a); });
  const COL = [NB.c.blue, NB.c.orange, NB.c.green], CNAME = ["blue", "orange", "green"];

  function trace(method) {
    const asg = new Array(N).fill(-1), dom = Array.from({ length: N }, () => [0, 1, 2]);
    const ev = [], cnt = { assign: 0, back: 0 };
    const snap = (kind, v, c, extra) => ev.push({ kind, v, c, asg: asg.slice(), dom: dom.map((d) => d.slice()), ...extra });
    snap("start", -1, -1);
    const pick = () => {
      const un = [...Array(N).keys()].filter((v) => asg[v] < 0);
      if (method !== "mrv") return un[0];
      const deg = (v) => nb[v].filter((u) => asg[u] < 0).length;
      return un.reduce((b, v) => (dom[v].length < dom[b].length || (dom[v].length === dom[b].length && deg(v) > deg(b)) ? v : b));
    };
    function rec() {
      if (asg.every((c) => c >= 0)) return true;
      const v = pick();
      for (const c of dom[v].slice()) {
        if (method === "bt" && nb[v].some((u) => asg[u] === c)) continue;
        cnt.assign++;
        asg[v] = c;
        const saved = dom.map((d) => d.slice());
        let wiped = -1;
        if (method !== "bt") {
          dom[v] = [c];
          for (const u of nb[v]) if (asg[u] < 0) { dom[u] = dom[u].filter((x) => x !== c); if (!dom[u].length && wiped < 0) wiped = u; }
        }
        snap("assign", v, c, { wiped });
        if (wiped < 0 && rec()) return true;
        asg[v] = -1;
        for (let u = 0; u < N; u++) dom[u] = saved[u];
      }
      cnt.back++;
      snap("back", v, -1);
      return false;
    }
    rec();
    snap("done", -1, -1);
    return { ev, cnt };
  }
  const runs = { bt: trace("bt"), fc: trace("fc"), mrv: trace("mrv") };

  const st = { method: P.method || "bt", step: P.step ?? -1 };
  const ctl = NB.controls(el);
  NB.select(ctl, "solver", [["bt", "plain backtracking"], ["fc", "forward checking"], ["mrv", "forward checking + MRV"]], st.method, (v) => {
    st.method = v; slider.input.max = runs[v].ev.length - 1; st.step = runs[v].ev.length - 1; slider.set(st.step); draw();
  });
  const slider = NB.slider(ctl, { label: "step", min: 0, max: runs[st.method].ev.length - 1, step: 1, value: 0, format: (v) => v.toFixed(0) }, (v) => { st.step = v; draw(); });
  if (st.step < 0) st.step = runs[st.method].ev.length - 1;
  slider.set(st.step);

  const W = 520, H = 330, svg = NB.svg(el, W, H);
  Object.assign(svg.style, { maxWidth: "560px", display: "block", margin: "0 auto" });
  const read = NB.readout(el);
  const X = (i) => 70 + pos[i][0] * 127, Y = (i) => 50 + pos[i][1] * 115;

  function draw() {
    NB.clear(svg);
    const { ev, cnt } = runs[st.method], e = ev[Math.min(st.step, ev.length - 1)];
    const showDom = st.method !== "bt";
    E.forEach(([a, b]) => NB.el("line", { x1: X(a), y1: Y(a), x2: X(b), y2: Y(b), stroke: NB.c.muted, "stroke-width": 2 }, svg));
    for (let i = 0; i < N; i++) {
      const c = e.asg[i], cur = i === e.v, wiped = e.wiped === i;
      NB.el("circle", {
        cx: X(i), cy: Y(i), r: 25, fill: c >= 0 ? COL[c] : NB.c.bg, "fill-opacity": c >= 0 ? 0.85 : 1,
        stroke: wiped || (cur && e.kind === "back") ? NB.c.pink : cur ? NB.c.fg : NB.c.muted, "stroke-width": wiped || cur ? 4 : 1.5,
        "stroke-dasharray": e.kind === "back" && cur ? "5 3" : null,
      }, svg);
      NB.el("text", { x: X(i), y: Y(i) + 6, "text-anchor": "middle", "font-size": 17, "font-weight": 600 }, svg, "X" + (i + 1));
      if (showDom) {   // remaining domain as three dots under the node
        for (let k = 0; k < K; k++) {
          const has = e.dom[i].includes(k);
          NB.el("circle", { cx: X(i) - 14 + 14 * k, cy: Y(i) + 37, r: 5, fill: has ? COL[k] : "none", stroke: has ? COL[k] : NB.c.grid, "stroke-width": 1.2 }, svg);
        }
      }
    }
    let what;
    if (e.kind === "start") what = "nothing assigned yet";
    else if (e.kind === "done") what = "every variable is assigned: a solution";
    else if (e.kind === "back") what = `no value of <b>X${e.v + 1}</b> works: back up to the previous variable`;
    else what = `assign <b>X${e.v + 1} = ${CNAME[e.c]}</b>` + (e.wiped >= 0 ? `: X${e.wiped + 1} has no values left, so undo at once` : "");
    const tot = (m) => `${runs[m].cnt.assign} assignments, ${runs[m].cnt.back} backtracks`;
    read.innerHTML = `step ${Math.min(st.step, ev.length - 1)} of ${ev.length - 1}: ${what}.<br>` +
      `plain backtracking: <b>${tot("bt")}</b>; forward checking: <b>${tot("fc")}</b>; with MRV: <b>${tot("mrv")}</b>.`;
  }
  draw();
});

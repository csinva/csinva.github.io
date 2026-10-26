/* nfa-run: an NFA over {0,1} that accepts strings ending in 01, run on a string symbol by symbol.
 * The highlighted states are the live copies of the machine; the set of them is the state of the
 * equivalent DFA (subset construction). Type a string or add symbols with the buttons.
 * params: {input: "1001"} */
NB.register("nfa-run", function (el, P) {
  const delta = { q0: { 0: ["q0", "q1"], 1: ["q0"] }, q1: { 0: [], 1: ["q2"] }, q2: { 0: [], 1: [] } };
  const accept = new Set(["q2"]);
  let input = (P.input === undefined ? "1001" : String(P.input)).replace(/[^01]/g, ""), t = input.length;
  const ctl = NB.controls(el);
  const lab = NB.el("label", { class: "nb-select" }, ctl);
  NB.el("span", { class: "nb-slider-label" }, lab, "input");
  const inp = NB.el("input", { type: "text", value: input, size: 14, style: { background: "var(--nb-panel)", color: "var(--nb-fg)", border: "1px solid var(--nb-border)", borderRadius: "6px", padding: "0.15rem 0.4rem", fontFamily: "var(--pst-font-family-monospace)" } }, lab);
  inp.addEventListener("input", () => { input = inp.value.replace(/[^01]/g, "").slice(0, 16); t = input.length; tS.input.max = input.length; tS.set(t); draw(); });
  NB.button(ctl, "+0", () => add("0")); NB.button(ctl, "+1", () => add("1"));
  const tS = NB.slider(ctl, { label: "symbols read", min: 0, max: input.length, step: 1, value: t, format: (v) => v.toFixed(0) }, (v) => { t = v; draw(); });
  function add(s) { if (input.length >= 16) return; input += s; inp.value = input; t = input.length; tS.input.max = input.length; tS.set(t); draw(); }
  const W = 640, H = 196;
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const pos = { q0: [130, 92], q1: [320, 92], q2: [510, 92] };

  function run(k) {                                           // sets of states after 0..k symbols
    let cur = new Set(["q0"]); const hist = [cur];
    for (let i = 0; i < k; i++) { const nx = new Set(); cur.forEach((q) => delta[q][input[i]].forEach((r) => nx.add(r))); cur = nx; hist.push(cur); }
    return hist;
  }

  function draw() {
    NB.clear(svg);
    const hist = run(t), cur = hist[t];
    const defs = NB.el("defs", {}, svg), mk = NB.el("marker", { id: "nfa-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    NB.el("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--nb-faint)" }, mk);
    const edge = (d, lx, ly, label) => { NB.el("path", { d, fill: "none", stroke: "var(--nb-faint)", "stroke-width": 1.5, "marker-end": "url(#nfa-arrow)" }, svg); NB.el("text", { x: lx, y: ly, "text-anchor": "middle", style: { fontSize: "13px", fill: "var(--nb-muted)" } }, svg, label); };
    edge("M40,92 L100,92", 70, 84, "start");
    edge("M112,68 C95,22 165,22 148,68", 130, 22, "0, 1");
    edge("M160,92 L290,92", 225, 84, "0");
    edge("M350,92 L480,92", 415, 84, "1");
    Object.entries(pos).forEach(([q, [x, y]]) => {
      const on = cur.has(q);
      NB.el("circle", { cx: x, cy: y, r: 28, fill: on ? "color-mix(in srgb, var(--nb-c-orange) 30%, var(--nb-bg))" : "var(--nb-panel)", stroke: on ? NB.c.orange : "var(--nb-faint)", "stroke-width": on ? 2.5 : 1.5 }, svg);
      if (accept.has(q)) NB.el("circle", { cx: x, cy: y, r: 23, fill: "none", stroke: on ? NB.c.orange : "var(--nb-faint)", "stroke-width": 1.2 }, svg);
      NB.el("text", { x, y: y + 5, "text-anchor": "middle", style: { fontSize: "14px", fontWeight: 600 } }, svg, q);
    });
    // the input tape
    const cw = 26, x0 = W / 2 - (Math.max(input.length, 1) * cw) / 2;
    [...input].forEach((s, i) => {
      NB.el("rect", { x: x0 + i * cw, y: 150, width: cw - 2, height: 26, rx: 3, fill: i < t ? "var(--nb-panel)" : "none", stroke: i === t - 1 ? NB.c.orange : "var(--nb-border)", "stroke-width": i === t - 1 ? 2 : 1 }, svg);
      NB.el("text", { x: x0 + i * cw + cw / 2 - 1, y: 168, "text-anchor": "middle", style: { fontSize: "14px", fill: i < t ? "var(--nb-fg)" : "var(--nb-faint)", fontFamily: "var(--pst-font-family-monospace)" } }, svg, s);
    });
    const fmt = (S) => (S.size ? "{" + [...S].sort().join(", ") + "}" : "∅ (every copy died)");
    const acc = [...cur].some((q) => accept.has(q));
    read.innerHTML = `after "${input.slice(0, t)}": live states <b>${fmt(cur)}</b>. ` +
      (t === input.length ? (acc ? "A copy is in the accept state, so the NFA <b>accepts</b>." : "No copy is in the accept state, so the NFA <b>rejects</b>.") : "");
  }
  draw();
});

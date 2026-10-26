/* page-replacement: run FIFO, LRU or OPT on a reference string with a given number of frames.
 * Top: frame contents after each reference, faults marked. Bottom: page faults vs number of frames for all
 * three algorithms (FIFO's curve can go up: Belady's anomaly).
 * params: {algo: "fifo" | "lru" | "opt", frames: 3, string: "textbook" | "belady"} */
NB.register("page-replacement", function (el, P) {
  const strings = {
    textbook: [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1],
    belady: [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
  };
  const st = { algo: P.algo || "fifo", frames: P.frames || 3, str: P.string || "belady" };
  const ctl = NB.controls(el);
  NB.select(ctl, "algorithm", [["fifo", "FIFO"], ["lru", "LRU"], ["opt", "optimal (OPT)"]], st.algo, (v) => { st.algo = v; draw(); });
  NB.slider(ctl, { label: "frames", min: 1, max: 6, step: 1, value: st.frames, format: (v) => v.toFixed(0) }, (v) => { st.frames = v; draw(); });
  NB.select(ctl, "reference string", [["belady", "1 2 3 4 1 2 5 1 2 3 4 5"], ["textbook", "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1"]], st.str, (v) => { st.str = v; draw(); });
  const W = 640, H1 = 200, H2 = 170;
  const svg = NB.svg(el, W, H1);
  const svg2 = NB.svg(el, W, H2);
  NB.legend(el, [["FIFO", NB.c.orange], ["LRU", NB.c.blue], ["OPT", NB.c.green]]);
  const read = NB.readout(el);
  const colOf = { fifo: NB.c.orange, lru: NB.c.blue, opt: NB.c.green };

  // returns the frame contents after each reference and whether it faulted
  function simulate(ref, nf, algo) {
    const frames = [], meta = [], steps = [];
    ref.forEach((p, t) => {
      let k = frames.indexOf(p), fault = false;
      if (k >= 0) { if (algo === "lru") meta[k] = t; }
      else {
        fault = true;
        if (frames.length < nf) { frames.push(p); meta.push(t); k = frames.length - 1; }
        else {
          let v = 0;
          if (algo === "opt") {                                // evict the page used furthest in the future (or never)
            let far = -1;
            frames.forEach((q, i) => { const nx = ref.indexOf(q, t + 1); const d = nx < 0 ? Infinity : nx; if (d > far) { far = d; v = i; } });
          } else v = meta.indexOf(Math.min(...meta));          // FIFO: oldest load time; LRU: oldest use time
          frames[v] = p; meta[v] = t; k = v;
        }
      }
      steps.push({ frames: frames.slice(), fault, slot: k });
    });
    return steps;
  }

  function draw() {
    NB.clear(svg); NB.clear(svg2);
    const ref = strings[st.str], n = ref.length, cw = Math.min(30, (W - 70) / n), ch = 22, x0 = 60, y0 = 26;
    const steps = simulate(ref, st.frames, st.algo);
    NB.el("text", { x: x0 - 8, y: y0 - 8, class: "nb-label", "text-anchor": "end" }, svg, "page");
    ref.forEach((p, t) => NB.el("text", { x: x0 + t * cw + cw / 2, y: y0 - 8, "text-anchor": "middle", style: { fontSize: "13px", fontWeight: 600 } }, svg, p));
    for (let f = 0; f < st.frames; f++) NB.el("text", { x: x0 - 8, y: y0 + f * ch + 15, class: "nb-tick", "text-anchor": "end" }, svg, `frame ${f + 1}`);
    steps.forEach((s, t) => {
      for (let f = 0; f < st.frames; f++) {
        const isNew = s.fault && s.slot === f;
        NB.el("rect", { x: x0 + t * cw + 1, y: y0 + f * ch + 1, width: cw - 2, height: ch - 2, rx: 3,
          fill: isNew ? colOf[st.algo] : "var(--nb-panel)", "fill-opacity": isNew ? 0.3 : 1, stroke: isNew ? colOf[st.algo] : "var(--nb-border)" }, svg);
        if (s.frames[f] !== undefined) NB.el("text", { x: x0 + t * cw + cw / 2, y: y0 + f * ch + 15, "text-anchor": "middle", style: { fontSize: "12px", fill: isNew ? "var(--nb-fg)" : "var(--nb-muted)" } }, svg, s.frames[f]);
      }
      if (s.fault) NB.el("text", { x: x0 + t * cw + cw / 2, y: y0 + st.frames * ch + 14, "text-anchor": "middle", style: { fontSize: "12px", fill: colOf[st.algo], fontWeight: 600 } }, svg, "×");
    });
    NB.el("text", { x: x0 - 8, y: y0 + st.frames * ch + 14, class: "nb-tick", "text-anchor": "end" }, svg, "fault");
    svg.setAttribute("viewBox", `0 0 ${W} ${Math.max(120, y0 + st.frames * ch + 24)}`);

    // faults vs frames for all three algorithms
    const m = { l: 44, r: 12, t: 10, b: 34 };
    const X = NB.scale(1, 6, m.l, W - m.r), Y = NB.scale(0, n, H2 - m.b, m.t);
    NB.axes(svg2, X, Y, { xlabel: "number of frames", ylabel: "page faults", xticks: 6, yticks: 4 });
    const curves = {};
    ["fifo", "lru", "opt"].forEach((a, k) => {
      const pts = [1, 2, 3, 4, 5, 6].map((f) => [f, simulate(ref, f, a).filter((s) => s.fault).length]);
      curves[a] = pts;
      const off = (k - 1) * 0.04;
      NB.el("path", { d: NB.line(pts.map(([f, v]) => [f + off, v]), X, Y), fill: "none", stroke: colOf[a], "stroke-width": a === st.algo ? 2.6 : 1.4, "stroke-opacity": a === st.algo ? 1 : 0.6 }, svg2);
      pts.forEach(([f, v]) => NB.el("circle", { cx: X(f + off), cy: Y(v), r: a === st.algo && f === st.frames ? 5 : 2.5, fill: colOf[a] }, svg2));
    });
    const f = (a) => curves[a][st.frames - 1][1];
    let txt = `${st.frames} frame${st.frames > 1 ? "s" : ""}: FIFO <b>${f("fifo")}</b>, LRU <b>${f("lru")}</b>, OPT <b>${f("opt")}</b> faults on ${n} references.`;
    const bel = curves.fifo.findIndex((p, i) => i > 0 && p[1] > curves.fifo[i - 1][1]);
    if (bel > 0) txt += ` FIFO gets <b>more</b> faults with ${bel + 1} frames than with ${bel} (Belady's anomaly).`;
    read.innerHTML = txt;
  }
  draw();
});

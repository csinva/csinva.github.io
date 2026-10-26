/* hash-probing: insert the same keys into a hash table of prime size m with separate chaining or one of the
 * three open-addressing schemes. Cells are shaded by how many probes the key needed; the arcs show the probe
 * sequence of the newest key. The readout gives the load factor and the average probes per successful find.
 * params: {method: "linear" | "quadratic" | "double" | "chaining", n: 20, m: 31} */
NB.register("hash-probing", function (el, P) {
  const m = P.m || 31;
  const u = NB.rng(5);
  const keys = Array.from({ length: 2 * m }, () => Math.floor(u() * 1000));
  const h = (k) => k % m, h2 = (k) => 7 - (k % 7);               // hash2 is never 0
  const st = { method: P.method || "linear", n: P.n || 20 };
  const maxN = () => (st.method === "chaining" ? 2 * m : m);
  const ctl = NB.controls(el);
  NB.select(ctl, "collisions", [["chaining", "separate chaining"], ["linear", "linear probing, f(i) = i"],
    ["quadratic", "quadratic probing, f(i) = i²"], ["double", "double hashing, f(i) = i·hash2(k)"]], st.method,
    (v) => { st.method = v; st.n = Math.min(st.n, maxN()); nS.input.max = maxN(); nS.set(st.n); draw(); });
  const nS = NB.slider(ctl, { label: "keys inserted, n", min: 1, max: maxN(), step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  const W = 640, cw = (W - 20) / m, top = 78, H = 200;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["found on the 1st probe", shade(1)], ["needs more probes (stronger color = more)", shade(4)], ["probe sequence of the newest key", NB.c.pink]]);
  const read = NB.readout(el);

  function shade(p) { // 1 probe: light blue; more probes: orange, darker with more
    return p <= 1 ? "color-mix(in srgb, var(--nb-c-blue) 40%, transparent)"
      : `color-mix(in srgb, var(--nb-c-orange) ${Math.min(100, 30 + 14 * p)}%, transparent)`;
  }

  function insertAll() {
    const table = new Array(m).fill(null), chains = Array.from({ length: m }, () => []), probes = [];
    let failed = null, lastSeq = [], lastKey = null;
    for (let t = 0; t < st.n; t++) {
      const k = keys[t];
      lastKey = k;
      if (st.method === "chaining") { chains[h(k)].push(k); probes.push(chains[h(k)].length); lastSeq = [h(k)]; continue; }
      const seq = [];
      let placed = false;
      for (let i = 0; i < m; i++) {
        const f = st.method === "linear" ? i : st.method === "quadratic" ? i * i : i * h2(k);
        const p = (h(k) + f) % m;
        seq.push(p);
        if (table[p] === null) { table[p] = { k, probes: i + 1 }; probes.push(i + 1); placed = true; break; }
      }
      lastSeq = seq;
      if (!placed) { failed = { k, tries: seq.length }; break; }
    }
    return { table, chains, probes, failed, lastSeq, lastKey };
  }

  function draw() {
    NB.clear(svg);
    const R = insertAll(), x = (i) => 10 + i * cw;
    for (let i = 0; i < m; i++) {
      const c = R.table[i], chain = R.chains[i];
      const fill = st.method === "chaining" ? (chain.length ? shade(1) : "var(--nb-panel)") : (c ? shade(c.probes) : "var(--nb-panel)");
      const r = NB.el("rect", { x: x(i) + 1, y: top, width: cw - 2, height: 24, rx: 3, fill, stroke: "var(--nb-border)" }, svg);
      if (c) NB.el("title", {}, r, `slot ${i}: key ${c.k}, ${c.probes} probe${c.probes > 1 ? "s" : ""}`);
      if (chain.length) NB.el("title", {}, r, `slot ${i}: keys ${chain.join(", ")}`);
      if (i % 5 === 0) NB.el("text", { x: x(i) + cw / 2, y: top - 8, class: "nb-tick", "text-anchor": "middle" }, svg, i);
      chain.slice(1).forEach((k, j) => NB.el("rect", { x: x(i) + 3, y: top + 28 + j * 12, width: cw - 6, height: 9, rx: 2, fill: shade(j + 2) }, svg));
    }
    const seq = R.lastSeq;
    // fixed height per method so the page doesn't jump while the slider moves
    const counts = new Array(m).fill(0); keys.forEach((k) => counts[h(k)]++);
    const bottom = st.method === "chaining" ? top + 34 + (Math.max(...counts) - 1) * 12 : top + 80;
    if (st.method !== "chaining") {
      // probe sequence of the newest key: arcs below the table between consecutive probes
      const yb = top + 26;
      for (let i = 1; i < seq.length; i++) {
        const a = x(seq[i - 1]) + cw / 2, b = x(seq[i]) + cw / 2, hgt = Math.min(80, 10 + Math.abs(b - a) * 0.22);
        NB.el("path", { d: `M${a},${yb} Q${(a + b) / 2},${yb + hgt} ${b},${yb}`, fill: "none", stroke: NB.c.pink, "stroke-width": 1.6 }, svg);
      }
      NB.el("circle", { cx: x(seq[0]) + cw / 2, cy: yb + 3, r: 3.5, fill: NB.c.pink }, svg);
      NB.el("text", { x: 10, y: 22, class: "nb-label" }, svg, `newest key ${R.lastKey}: hash(k) = ${R.lastKey} mod ${m} = ${seq[0]}, ${R.failed ? "no free slot after" : "placed after"} ${seq.length} probe${seq.length > 1 ? "s" : ""}`);
    } else {
      NB.el("text", { x: 10, y: 22, class: "nb-label" }, svg, `newest key ${R.lastKey}: hash(k) = ${R.lastKey} mod ${m} = ${seq[0]}, added to that slot's list`);
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${bottom}`);
    // longest run of occupied cells (a cluster), wrapping around the end of the table
    let run = 0, best = 0;
    if (st.method !== "chaining") for (let i = 0; i < 2 * m; i++) { run = R.table[i % m] ? run + 1 : 0; best = Math.max(best, Math.min(run, m)); }
    const n = R.probes.length, lam = n / m, avg = R.probes.reduce((s, p) => s + p, 0) / Math.max(1, n);
    let txt = `load factor λ = ${n}/${m} = <b>${NB.fmt(lam, 2)}</b>; average probes per successful find <b>${NB.fmt(avg, 2)}</b>`;
    if (st.method === "chaining") txt += ` (the note's 1 + λ/2 = ${NB.fmt(1 + lam / 2, 2)})`;
    else txt += `; longest cluster <b>${best}</b> slots`;
    if (R.failed) txt += `. <b>Inserting key ${R.failed.k} failed</b> after ${R.failed.tries} probes: time to rehash into a bigger table.`;
    read.innerHTML = txt;
  }
  draw();
});

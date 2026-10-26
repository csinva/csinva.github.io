/* ngram-backoff: an unbounded n-gram model with backoff on a tiny corpus, the way infini-gram picks n.
 * For a prompt, try its suffixes from longest to shortest and use the longest one that occurs in the corpus
 * (with a token after it). The next-token distribution is count(suffix + w) / count(suffix). The corpus is shown
 * with every match of the chosen suffix underlined and the token that follows it highlighted. Click a bar to
 * append that token to the prompt.
 * params: {prompt: "my dog sat on the"} */
NB.register("ngram-backoff", function (el, P) {
  const CORPUS = "the cat sat on the mat . the dog sat on the rug . the cat ate the fish . the dog ate my shoe . a cat sat on my lap .";
  const T = CORPUS.split(" ");
  const PRESETS = ["my dog sat on the", "the cat sat on", "a bird ate", "my cat", "a bird"];
  const st = { prompt: (P.prompt || PRESETS[0]).split(" ") };

  function matches(suf) {   // start positions of suf in T that have a following token
    const out = [];
    for (let i = 0; i + suf.length < T.length; i++) if (suf.every((w, j) => T[i + j] === w)) out.push(i);
    return out;
  }
  function model(prompt) {
    const rows = [];
    let used = null;
    for (let L = prompt.length; L >= 0; L--) {
      const suf = prompt.slice(prompt.length - L), pos = matches(suf);
      rows.push({ suf, L, n: pos.length });
      if (pos.length && !used) used = { suf, L, pos };
    }
    const counts = {};
    used.pos.forEach((i) => (counts[T[i + used.L]] = (counts[T[i + used.L]] || 0) + 1));
    const dist = Object.entries(counts).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
    return { rows, used, dist, total: used.pos.length };
  }

  const ctl = NB.controls(el);
  const sel = NB.select(ctl, "start from", PRESETS.map((p) => [p, "“" + p + "”"]), PRESETS.includes(P.prompt) ? P.prompt : PRESETS[0], (v) => { st.prompt = v.split(" "); draw(); });
  NB.button(ctl, "drop last word", () => { if (st.prompt.length > 1) { st.prompt.pop(); draw(); } });
  const box = NB.el("div", { style: { fontSize: "0.95em", lineHeight: "1.6", margin: "0.4em 0" } }, el);
  const svg = NB.svg(el, 520, 230);
  Object.assign(svg.style, { maxWidth: "560px", display: "block", margin: "0 auto" });
  const corp = NB.el("div", { style: { fontSize: "0.9em", lineHeight: "1.9", marginTop: "0.4em", color: "var(--nb-fg)" } }, el);
  const read = NB.readout(el);
  const blue = "var(--nb-c-blue)", orange = "var(--nb-c-orange)", muted = "var(--nb-muted)";

  function draw() {
    const { rows, used, dist, total } = model(st.prompt);
    NB.clear(box);
    const pl = NB.el("div", {}, box);
    NB.el("span", { style: { color: muted } }, pl, "prompt: ");
    st.prompt.forEach((w, i) => {
      const inSuf = i >= st.prompt.length - used.L;
      NB.el("span", { style: { color: inSuf ? blue : "inherit", fontWeight: inSuf ? 700 : 400 } }, pl, w + " ");
    });
    const tb = NB.el("div", { style: { fontFamily: "var(--nb-mono, monospace)", fontSize: "0.85em", marginTop: "0.3em" } }, box);
    rows.forEach((r) => {
      const isUsed = r.L === used.L, line = NB.el("div", { style: { color: isUsed ? blue : r.n ? "inherit" : muted, fontWeight: isUsed ? 700 : 400 } }, tb);
      if (r.L < used.L) line.style.opacity = 0.45;
      line.textContent = `n = ${r.L + 1}  “${r.suf.join(" ") || "(empty context)"}” occurs ${r.n}×` + (isUsed ? "  ← longest match, use it" : r.L > used.L ? "  (no match, back off)" : "");
    });

    NB.clear(svg);
    const x0 = 120, x1 = 470, bh = dist.length > 6 ? 24 : 32, fs = dist.length > 6 ? 15 : 17, x = NB.scale(0, 1, x0, x1);
    dist.forEach(([w, c], i) => {
      const y = 15 + i * bh, g = NB.el("g", { style: { cursor: "pointer" } }, svg);
      NB.el("rect", { x: x0, y, width: Math.max(1, x(c / total) - x0), height: bh - 6, fill: NB.c.orange, "fill-opacity": 0.75, rx: 3 }, g);
      NB.el("rect", { x: 0, y, width: 520, height: bh - 6, fill: "transparent" }, g);
      NB.el("text", { x: x0 - 10, y: y + (bh - 6) / 2 + 5, "text-anchor": "end", "font-size": fs }, g, w);
      NB.el("text", { x: x(c / total) + 8, y: y + (bh - 6) / 2 + 5, "font-size": fs - 2, fill: NB.c.muted }, g, `${c}/${total} = ${NB.fmt(c / total, 2)}`);
      g.addEventListener("click", () => { st.prompt.push(w); draw(); });
    });
    svg.setAttribute("viewBox", `0 0 520 ${Math.ceil(25 + dist.length * bh)}`);

    NB.clear(corp);
    NB.el("span", { style: { color: muted } }, corp, "corpus: ");
    const mark = new Array(T.length).fill(0);
    used.pos.forEach((i) => { for (let j = 0; j < used.L; j++) mark[i + j] = 1; mark[i + used.L] = 2; });
    T.forEach((w, i) => {
      const sty = mark[i] === 1 ? { color: blue, textDecoration: "underline", fontWeight: 600 } : mark[i] === 2 ? { color: orange, fontWeight: 700 } : {};
      NB.el("span", { style: sty }, corp, w);
      corp.appendChild(document.createTextNode(" "));
    });
    read.innerHTML = `Backing off to n = <b>${used.L + 1}</b> (a context of ${used.L} token${used.L === 1 ? "" : "s"}). ` +
      `${total} match${total === 1 ? "" : "es"} in the corpus, so the next-token probabilities are counts out of ${total}. Click a bar to add that token.`;
  }
  draw();
});

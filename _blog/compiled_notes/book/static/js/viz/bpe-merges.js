/* bpe-merges: byte-pair encoding on a small corpus. Start from characters; each merge finds the most frequent
 * adjacent pair of tokens, makes it a new token, and re-tokenizes everything. "_" marks the end of a word.
 * params: {merges: 4, text: "..."} */
NB.register("bpe-merges", function (el, P) {
  const defaultText = P.text || "low low low low low lower lower newest newest newest newest newest newest widest widest widest";
  const st = { k: P.merges ?? 4, text: defaultText };
  const ctl = NB.controls(el);
  const slider = NB.slider(ctl, { label: "merges", min: 0, max: 15, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; draw(); });
  const box = NB.el("label", { class: "nb-select", style: { flex: "1 1 18rem" } }, ctl);
  NB.el("span", { class: "nb-slider-label" }, box, "corpus");
  const input = NB.el("input", { type: "text", value: st.text, style: { flex: "1", minWidth: "10rem", background: "var(--nb-panel)", color: "var(--nb-fg)", border: "1px solid var(--nb-border)", borderRadius: "6px", padding: "0.2rem 0.4rem", font: "inherit" } }, box);
  input.addEventListener("input", () => { st.text = input.value; draw(); });
  const out = NB.el("div", { style: { display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)", gap: "14px", marginTop: "0.4rem" } }, el);
  const wordsDiv = NB.el("div", {}, out), mergesDiv = NB.el("div", { style: { fontSize: "0.84rem" } }, out);
  const read = NB.readout(el);
  const chip = (parent, t, hot) => NB.el("span", { style: { display: "inline-block", fontFamily: "var(--pst-font-family-monospace)", fontSize: "0.85rem", padding: "0.05rem 0.3rem", margin: "1px", borderRadius: "4px",
    border: `1px solid ${hot ? "var(--nb-c-orange)" : "var(--nb-border)"}`, background: "var(--nb-panel)", color: hot ? "var(--nb-c-orange)" : "var(--nb-fg)" } }, parent, t);

  function bpe(text, k) {
    const counts = new Map();
    text.toLowerCase().split(/\s+/).filter(Boolean).forEach((w) => counts.set(w, (counts.get(w) || 0) + 1));
    let words = [...counts.entries()].map(([w, c]) => ({ toks: [...w, "_"], c }));
    const merges = [];
    for (let s = 0; s < k; s++) {
      const pc = new Map();   // pair -> count; Map keeps first-seen order, which breaks ties
      words.forEach(({ toks, c }) => { for (let i = 0; i + 1 < toks.length; i++) { const key = toks[i] + "\u0000" + toks[i + 1]; pc.set(key, (pc.get(key) || 0) + c); } });
      let best = null, bc = 1;
      pc.forEach((c, key) => { if (c > bc) { bc = c; best = key; } });
      if (!best) break;
      const [a, b] = best.split("\u0000");
      merges.push([a, b, bc]);
      words = words.map(({ toks, c }) => { const t = []; for (let i = 0; i < toks.length; i++) { if (i + 1 < toks.length && toks[i] === a && toks[i + 1] === b) { t.push(a + b); i++; } else t.push(toks[i]); } return { toks: t, c }; });
    }
    return { words, merges };
  }
  function draw() {
    NB.clear(wordsDiv); NB.clear(mergesDiv);
    const { words, merges } = bpe(st.text, st.k), last = merges.length ? merges[merges.length - 1] : null;
    words.forEach(({ toks, c }) => {
      const line = NB.el("div", { style: { margin: "0.15rem 0" } }, wordsDiv);
      toks.forEach((t) => chip(line, t, last && t === last[0] + last[1]));
      NB.el("span", { style: { color: "var(--nb-muted)", fontSize: "0.8rem", marginLeft: "0.4rem" } }, line, `× ${c}`);
    });
    NB.el("div", { style: { color: "var(--nb-muted)", marginBottom: "0.2rem" } }, mergesDiv, "merges, in order (pair count)");
    if (!merges.length) NB.el("div", { style: { color: "var(--nb-muted)" } }, mergesDiv, "none yet: every token is a character");
    merges.forEach(([a, b, c], i) => NB.el("div", { style: { fontFamily: "var(--pst-font-family-monospace)", color: i === merges.length - 1 ? "var(--nb-c-orange)" : "var(--nb-fg)" } }, mergesDiv, `${i + 1}. ${a} + ${b} → ${a + b}  (${c})`));
    const vocab = new Set(), chars = new Set();
    let total = 0;
    words.forEach(({ toks, c }) => { toks.forEach((t) => vocab.add(t)); total += toks.length * c; });
    st.text.toLowerCase().replace(/\s+/g, "").split("").forEach((ch) => chars.add(ch));
    read.innerHTML = `${chars.size + 1} characters (with _) + ${merges.length} merges · tokens in use <b>${vocab.size}</b> · corpus length <b>${total}</b> tokens` + (merges.length < st.k ? " · no pair occurs twice, so merging stops" : "");
  }
  draw();
});

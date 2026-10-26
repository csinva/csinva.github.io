/* float-bits: the 32 bits of an IEEE single-precision float. Click a bit to flip it, or type a number.
 * Shows sign / exponent / mantissa, the decoded value, the hex pattern, and the gap to the next float.
 * params: {value: 42.125} */
NB.register("float-bits", function (el, P) {
  const buf = new DataView(new ArrayBuffer(4));
  let bits = 0;
  const setFloat = (x) => { buf.setFloat32(0, x); bits = buf.getUint32(0); };
  const getFloat = (b) => { buf.setUint32(0, b >>> 0); return buf.getFloat32(0); };
  setFloat(P.value === undefined ? 42.125 : P.value);

  const ctl = NB.controls(el);
  const lab = NB.el("label", { class: "nb-select" }, ctl);
  NB.el("span", { class: "nb-slider-label" }, lab, "number");
  const inp = NB.el("input", { type: "text", value: String(P.value === undefined ? 42.125 : P.value), size: 12,
    style: { background: "var(--nb-panel)", color: "var(--nb-fg)", border: "1px solid var(--nb-border)", borderRadius: "6px", padding: "0.15rem 0.4rem", font: "inherit" } }, lab);
  inp.addEventListener("change", () => { const v = Number(inp.value); if (!isNaN(v) || inp.value.trim().toLowerCase() === "nan") { setFloat(isNaN(v) ? NaN : v); draw(false); } });
  const presets = [["42.125", 42.125], ["0.1", 0.1], ["1", 1], ["largest", getFloat(0x7f7fffff)], ["smallest normal", getFloat(0x00800000)], ["16777217", 16777217]];
  presets.forEach(([t, v]) => NB.button(ctl, t, () => { setFloat(v); draw(true); }));

  const W = 640, H = 128, bw = (W - 20) / 32, y0 = 30;
  const svg = NB.svg(el, W, H);
  const read = NB.readout(el);
  const fields = [[0, 1, NB.c.pink, "sign"], [1, 9, NB.c.blue, "exponent (8 bits)"], [9, 32, NB.c.green, "mantissa (23 bits)"]];

  function draw(updateInput) {
    NB.clear(svg);
    fields.forEach(([a, b, col, name]) => {
      NB.el("text", { x: 10 + a * bw + (b - a) * bw / 2, y: 16, class: "nb-label", "text-anchor": "middle", style: { fill: col } }, svg, name);
      for (let i = a; i < b; i++) {
        const on = (bits >>> (31 - i)) & 1;
        const g = NB.el("g", { style: { cursor: "pointer" } }, svg);
        NB.el("rect", { x: 10 + i * bw + 1, y: y0, width: bw - 2, height: 30, rx: 3, fill: on ? col : "var(--nb-panel)", "fill-opacity": on ? 0.85 : 1, stroke: col, "stroke-opacity": 0.6 }, g);
        NB.el("text", { x: 10 + i * bw + bw / 2, y: y0 + 20, "text-anchor": "middle", style: { fill: on ? "var(--nb-bg)" : "var(--nb-muted)", fontSize: "13px", fontWeight: 600 } }, g, on);
        g.addEventListener("click", () => { bits = (bits ^ (1 << (31 - i))) >>> 0; draw(true); });
      }
    });
    const s = bits >>> 31, e = (bits >>> 23) & 0xff, man = bits & 0x7fffff, x = getFloat(bits);
    const frac = man / 2 ** 23;
    let formula;
    if (e === 255) formula = man ? "exponent 255 and mantissa ≠ 0: NaN" : `exponent 255 and mantissa 0: ${s ? "−" : "+"}infinity`;
    else if (e === 0) formula = `exponent 0 (denormalized): (−1)^${s} × ${NB.fmt(frac, 6)} × 2^−126`;
    else formula = `(−1)^${s} × (1 + ${frac.toPrecision(10).replace(/\.?0+$/, "")}) × 2^(${e} − 127) = (−1)^${s} × ${(1 + frac).toPrecision(10).replace(/\.?0+$/, "")} × 2^${e - 127}`;
    NB.el("text", { x: 10, y: y0 + 58, style: { fontSize: "13px" } }, svg, formula);
    NB.el("text", { x: 10, y: y0 + 82, style: { fontSize: "13px", fill: "var(--nb-muted)" } }, svg, `hex 0x${bits.toString(16).padStart(8, "0")}`);
    const next = isFinite(x) ? getFloat(s === 0 ? (bits + 1) >>> 0 : bits === 0x80000000 ? 1 : (bits - 1) >>> 0) - x : NaN;
    const shown = isFinite(x) ? (Math.abs(x) >= 1e10 || (Math.abs(x) < 1e-4 && x !== 0) ? x.toExponential(8) : x.toPrecision(17).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "")) : String(x);
    read.innerHTML = `stored value <b>${shown}</b>` + (isFinite(next) ? `; the next float up is <b>${next.toExponential(2)}</b> away` : "");
    if (updateInput) inp.value = isFinite(x) ? String(+x.toPrecision(9)) : String(x);
  }
  draw(false);
});

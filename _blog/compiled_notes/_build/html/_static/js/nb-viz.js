/* nb-viz: tiny framework for the book's interactive figures.
 *
 * A page includes   <div class="nb-viz" data-viz="NAME" data-params='{"k": 1}'></div>
 * (the build inserts these from the overlays). Widgets register themselves with
 *   NB.register("NAME", function (el, params) { ... })
 * and draw with the helpers below. Colors come from CSS variables (--nb-c-*) so every widget
 * follows the light/dark theme without re-rendering. Keep widgets dependency-free; d3 v7 is
 * loaded on the page if a widget really needs it (window.d3).
 */
(function () {
  const registry = {};
  const NB = (window.NB = {});

  NB.register = function (name, fn) { registry[name] = fn; };

  NB.colors = ["var(--nb-c-blue)", "var(--nb-c-orange)", "var(--nb-c-green)", "var(--nb-c-pink)",
               "var(--nb-c-purple)", "var(--nb-c-yellow)"];
  NB.c = { blue: "var(--nb-c-blue)", orange: "var(--nb-c-orange)", green: "var(--nb-c-green)",
           pink: "var(--nb-c-pink)", purple: "var(--nb-c-purple)", yellow: "var(--nb-c-yellow)",
           fg: "var(--nb-fg)", muted: "var(--nb-muted)", grid: "var(--nb-grid)", bg: "var(--nb-bg)" };

  // ---- DOM / SVG helpers -------------------------------------------------------------
  const SVGNS = "http://www.w3.org/2000/svg";
  NB.el = function (tag, attrs, parent, text) {
    const svgTags = /^(svg|g|path|line|circle|rect|text|polyline|polygon|ellipse|defs|marker|clipPath|tspan)$/;
    const inSvg = svgTags.test(tag) || (parent && parent.namespaceURI === SVGNS);   // e.g. <a> inside an svg
    const e = inSvg ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
    for (const k in attrs || {}) {
      if (k === "style" && typeof attrs[k] === "object") Object.assign(e.style, attrs[k]);
      else if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    }
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };

  /** Responsive svg: drawn in a fixed w x h coordinate system, scaled to the container width. */
  NB.svg = function (parent, w, h, cls) {
    return NB.el("svg", { viewBox: `0 0 ${w} ${h}`, class: "nb-svg " + (cls || ""), preserveAspectRatio: "xMidYMid meet" }, parent);
  };

  NB.clear = function (node) { while (node.firstChild) node.removeChild(node.firstChild); };

  /** Linear scale: maps [d0, d1] -> [r0, r1]; .inv() maps back. */
  NB.scale = function (d0, d1, r0, r1) {
    const f = (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);
    f.inv = (y) => d0 + ((y - r0) / (r1 - r0)) * (d1 - d0);
    f.domain = [d0, d1]; f.range = [r0, r1];
    return f;
  };

  NB.ticks = function (a, b, n) {
    const span = b - a, step0 = span / (n || 5), mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= (n || 5)) || mag * 10;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };

  NB.fmt = function (v, d) {
    if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
    return (+v.toFixed(d === undefined ? 2 : d)).toString();
  };

  /** Axes with ticks, light grid and labels. x, y are NB.scale objects. */
  NB.axes = function (g, x, y, opts) {
    opts = opts || {};
    const ax = NB.el("g", { class: "nb-axes" }, g);
    const [x0, x1] = x.range, [y0, y1] = y.range;
    if (opts.grid !== false) {
      NB.ticks(...x.domain, opts.xticks).forEach((t) => NB.el("line", { x1: x(t), x2: x(t), y1: y0, y2: y1, class: "nb-grid" }, ax));
      NB.ticks(...[y.domain[0], y.domain[1]].sort((a, b) => a - b), opts.yticks).forEach((t) => NB.el("line", { x1: x0, x2: x1, y1: y(t), y2: y(t), class: "nb-grid" }, ax));
    }
    NB.el("line", { x1: x0, x2: x1, y1: y0, y2: y0, class: "nb-axis" }, ax);
    NB.el("line", { x1: x0, x2: x0, y1: y0, y2: y1, class: "nb-axis" }, ax);
    NB.ticks(...x.domain, opts.xticks).forEach((t) => NB.el("text", { x: x(t), y: y0 + 16, class: "nb-tick", "text-anchor": "middle" }, ax, opts.xfmt ? opts.xfmt(t) : NB.fmt(t)));
    NB.ticks(...[y.domain[0], y.domain[1]].sort((a, b) => a - b), opts.yticks).forEach((t) => NB.el("text", { x: x0 - 6, y: y(t) + 4, class: "nb-tick", "text-anchor": "end" }, ax, opts.yfmt ? opts.yfmt(t) : NB.fmt(t)));
    if (opts.xlabel) NB.el("text", { x: (x0 + x1) / 2, y: y0 + 34, class: "nb-label", "text-anchor": "middle" }, ax, opts.xlabel);
    if (opts.ylabel) NB.el("text", { x: 0, y: 0, class: "nb-label", "text-anchor": "middle", transform: `translate(${x0 - 38},${(y0 + y1) / 2}) rotate(-90)` }, ax, opts.ylabel);
    return ax;
  };

  NB.line = function (pts, x, y) {
    return pts.filter((p) => isFinite(p[1])).map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join("");
  };

  NB.legend = function (parent, items) {
    const box = NB.el("div", { class: "nb-legend" }, parent);
    items.forEach(([label, color, dashed]) => {
      const it = NB.el("span", { class: "nb-legend-item" }, box);
      NB.el("span", { class: "nb-swatch" + (dashed ? " dashed" : ""), style: { background: dashed ? "none" : color, borderColor: color } }, it);
      NB.el("span", {}, it, label);
    });
    return box;
  };

  // ---- controls ----------------------------------------------------------------------
  NB.controls = function (parent) { return NB.el("div", { class: "nb-controls" }, parent); };

  /** Labeled range slider. onInput(value) fires on every move. Returns {input, set(v), get()}. */
  NB.slider = function (parent, o, onInput) {
    const wrap = NB.el("label", { class: "nb-slider" }, parent);
    NB.el("span", { class: "nb-slider-label" }, wrap, o.label);
    const input = NB.el("input", { type: "range", min: o.min, max: o.max, step: o.step || (o.max - o.min) / 100, value: o.value }, wrap);
    const out = NB.el("span", { class: "nb-slider-value" }, wrap);
    const show = () => (out.textContent = o.format ? o.format(+input.value) : NB.fmt(+input.value));
    input.addEventListener("input", () => { show(); onInput(+input.value); });
    show();
    return { input, get: () => +input.value, set: (v) => { input.value = v; show(); } };
  };

  NB.button = function (parent, label, onClick, cls) {
    const b = NB.el("button", { type: "button", class: "nb-button " + (cls || "") }, parent, label);
    b.addEventListener("click", onClick);
    return b;
  };

  NB.select = function (parent, label, options, value, onChange) {
    const wrap = NB.el("label", { class: "nb-select" }, parent);
    NB.el("span", { class: "nb-slider-label" }, wrap, label);
    const s = NB.el("select", {}, wrap);
    options.forEach(([v, t]) => { const o = NB.el("option", { value: v }, s, t); if (v === value) o.selected = true; });
    s.addEventListener("change", () => onChange(s.value));
    return s;
  };

  NB.readout = function (parent) { return NB.el("div", { class: "nb-readout" }, parent); };

  // ---- numerics ----------------------------------------------------------------------
  /** Seeded PRNG so figures look the same on every load. */
  NB.rng = function (seed) {
    let s = seed >>> 0 || 1;
    const u = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
    u.normal = () => { let a = 0, b = 0; while (!a) a = u(); while (!b) b = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); };
    return u;
  };

  NB.range = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

  /** Solve A x = b (small dense systems) by Gaussian elimination with partial pivoting. */
  NB.solve = function (A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) continue;
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r, i) => (Math.abs(r[i]) < 1e-12 ? 0 : r[n] / r[i]));
  };

  /** Least squares with an optional ridge penalty: argmin ||X w - y||^2 + lam ||w||^2. */
  NB.lstsq = function (X, y, lam) {
    const d = X[0].length;
    const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? lam || 0 : 0)));
    const b = Array.from({ length: d }, (_, i) => X.reduce((s, r, k) => s + r[i] * y[k], 0));
    return NB.solve(A, b);
  };

  // ---- boot ----------------------------------------------------------------------------
  function boot() {
    document.querySelectorAll(".nb-viz:not([data-ready])").forEach((el) => {
      const name = el.dataset.viz, fn = registry[name];
      el.dataset.ready = "1";
      if (!fn) { el.textContent = `(missing widget "${name}")`; el.classList.add("nb-viz-missing"); return; }
      let params = {};
      try { params = JSON.parse(el.dataset.params || "{}"); } catch (e) { console.warn("nb-viz params", name, e); }
      try { fn(el, params); } catch (e) { console.error("nb-viz", name, e); el.textContent = `(widget "${name}" failed: ${e.message})`; return; }
      // a small reset button: redraw the widget with the parameters it started with
      const fig = el.closest(".nb-viz-figure");
      if (fig && !fig.querySelector(".nb-reset") && name !== "topic-map") {
        const b = NB.el("button", { type: "button", class: "nb-reset", title: "Reset this figure", "aria-label": "Reset this figure" }, fig, "↺ reset");
        b.addEventListener("click", () => { NB.clear(el); try { fn(el, JSON.parse(el.dataset.params || "{}")); } catch (e) { console.error(e); } });
      }
    });
  }
  NB.boot = boot;

  // mermaid diagrams from the notes (```mermaid blocks): drawn in the neutral theme, inverted in dark mode by css
  function bootMermaid() {
    if (!window.mermaid || !document.querySelector("pre.mermaid")) return;
    window.mermaid.initialize({ startOnLoad: false, theme: "neutral", fontFamily: "Inter, sans-serif" });
    window.mermaid.run({ querySelector: "pre.mermaid" });
  }
  window.addEventListener("load", bootMermaid);

  // previous / next links at the top of each page too (copied from the theme's footer links)
  function topPrevNext() {
    const src = document.querySelector("footer .prev-next-area");
    const art = document.querySelector(".bd-article");
    if (!src || !art || document.querySelector(".top-prev-next")) return;
    const bar = document.createElement("nav");
    bar.className = "top-prev-next";
    bar.setAttribute("aria-label", "Previous and next page");
    src.querySelectorAll("a.left-prev, a.right-next").forEach((a) => {
      const prev = a.classList.contains("left-prev");
      const title = (a.querySelector(".prev-next-title") || {}).textContent || "";
      const link = document.createElement("a");
      link.href = a.getAttribute("href");
      link.className = prev ? "tpn-prev" : "tpn-next";
      link.title = (prev ? "Previous: " : "Next: ") + title;
      link.textContent = prev ? "‹ " + title : title + " ›";
      bar.appendChild(link);
    });
    if (bar.children.length) art.prepend(bar);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", topPrevNext); else topPrevNext();

  // glossary page: filter entries as you type
  document.addEventListener("input", (e) => {
    if (!e.target.classList || !e.target.classList.contains("glossary-filter")) return;
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll(".glossary-letter").forEach((sec) => {
      let any = false;
      sec.querySelectorAll(".glossary-entry").forEach((en) => {
        const hit = !q || en.textContent.toLowerCase().includes(q);
        en.style.display = hit ? "" : "none"; any = any || hit;
      });
      sec.style.display = any ? "" : "none";
    });
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => setTimeout(boot));
  else setTimeout(boot);
})();

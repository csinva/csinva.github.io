/* central-limit: histogram of sample means of n draws from a skewed, flat or two-humped population.
 * As n grows the histogram narrows (law of large numbers) and turns bell-shaped (central limit theorem).
 * params: {n: 1, population: "exponential" | "uniform" | "bimodal", reps: 4000} */
NB.register("central-limit", function (el, P) {
  const pops = {
    exponential: { label: "skewed (exponential)", draw: (u) => -Math.log(1 - u()), mu: 1, sd: 1, lo: 0, hi: 4 },
    uniform: { label: "flat (uniform)", draw: (u) => u(), mu: 0.5, sd: Math.sqrt(1 / 12), lo: 0, hi: 1 },
    bimodal: { label: "two humps", draw: (u) => (u() < 0.5 ? 0.2 : 0.8) + 0.07 * u.normal(), mu: 0.5, sd: Math.sqrt(0.09 + 0.0049), lo: 0, hi: 1 },
  };
  const st = { n: P.n || 1, pop: P.population || "exponential" }, reps = P.reps || 4000;
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "samples averaged, n", min: 1, max: 50, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  NB.select(ctl, "population", Object.entries(pops).map(([k, v]) => [k, v.label]), st.pop, (v) => { st.pop = v; draw(); });
  const W = 640, H = 300, m = { l: 44, r: 10, t: 10, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["sample means", NB.c.blue], ["normal curve predicted by the CLT", NB.c.orange]]);
  const read = NB.readout(el);
  function draw() {
    NB.clear(svg);
    const p = pops[st.pop], u = NB.rng(11 + st.n), bins = 48;
    const means = Array.from({ length: reps }, () => { let s = 0; for (let i = 0; i < st.n; i++) s += p.draw(u); return s / st.n; });
    const X = NB.scale(p.lo, p.hi, m.l, W - m.r), bw = (p.hi - p.lo) / bins;
    const counts = new Array(bins).fill(0);
    means.forEach((x) => { const b = Math.floor((x - p.lo) / bw); if (b >= 0 && b < bins) counts[b]++; });
    const dens = counts.map((c) => c / (reps * bw)), se = p.sd / Math.sqrt(st.n);
    const peak = Math.max(...dens, 1 / (se * Math.sqrt(2 * Math.PI)));
    const Y = NB.scale(0, peak * 1.1, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "value of the sample mean", ylabel: "density", xticks: 6, yticks: 4 });
    dens.forEach((d, i) => NB.el("rect", { x: X(p.lo + i * bw) + 0.5, y: Y(d), width: Math.max(0, X(p.lo + bw) - X(p.lo) - 1), height: Y(0) - Y(d), fill: NB.c.blue, "fill-opacity": 0.55 }, svg));
    const xs = NB.range(p.lo, p.hi, 200);
    NB.el("path", { d: NB.line(xs.map((x) => [x, Math.exp(-0.5 * ((x - p.mu) / se) ** 2) / (se * Math.sqrt(2 * Math.PI))]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg);
    const sd = Math.sqrt(means.reduce((s, x) => s + (x - p.mu) ** 2, 0) / reps);
    read.innerHTML = `n = ${st.n}: spread of the sample mean <b>${NB.fmt(sd, 3)}</b>, predicted σ/√n = <b>${NB.fmt(se, 3)}</b>`;
  }
  draw();
});

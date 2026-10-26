/* test-power: upper-tailed z-test of H0: θ = 0 vs H1: θ > 0 from the mean of n draws with σ = 1.
 * Draws the sampling distribution of the sample mean under H0 and under H1 (true θ), the rejection cutoff,
 * the type 1 error α (H0 area past the cutoff) and the type 2 error β (H1 area before it). Power = 1 - β.
 * params: {theta: 0.5, n: 10, alpha: 0.05} */
NB.register("test-power", function (el, P) {
  const Phi = (z) => {                                   // standard normal CDF (Abramowitz & Stegun 7.1.26)
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const e = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + e) : 0.5 * (1 - e);
  };
  const PhiInv = (p) => {                                // inverse normal CDF (Acklam)
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const q0 = Math.min(p, 1 - p);
    if (q0 < 0.02425) {
      const q = Math.sqrt(-2 * Math.log(q0));
      const x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
      return p < 0.5 ? x : -x;
    }
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  };
  const st = { th: P.theta ?? 0.5, n: P.n || 10, al: P.alpha || 0.05 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "true effect θ", min: 0, max: 1.5, step: 0.05, value: st.th }, (v) => { st.th = v; draw(); });
  NB.slider(ctl, { label: "sample size n", min: 1, max: 100, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  NB.slider(ctl, { label: "α", min: 0.01, max: 0.2, step: 0.01, value: st.al }, (v) => { st.al = v; draw(); });
  const W = 640, H = 290, m = { l: 44, r: 10, t: 14, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["sample mean under H0 (θ = 0)", NB.c.blue], ["under H1 (true θ)", NB.c.orange], ["α: reject H0 though it is true", NB.c.blue], ["β: keep H0 though it is false", NB.c.orange]]);
  const read = NB.readout(el);
  const X = NB.scale(-1.5, 2.5, m.l, W - m.r);
  function draw() {
    NB.clear(svg);
    const se = 1 / Math.sqrt(st.n), cut = PhiInv(1 - st.al) * se;
    const dens = (x, mu) => Math.exp(-0.5 * ((x - mu) / se) ** 2) / (se * Math.sqrt(2 * Math.PI));
    const Y = NB.scale(0, dens(0, 0) * 1.12, H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "sample mean x̄  (σ = 1)", ylabel: "density", xticks: 8, yticks: 4 });
    const xs = NB.range(-1.5, 2.5, 400);
    const area = (lo, hi, mu, color) => {
      const pts = xs.filter((x) => x >= lo && x <= hi);
      if (pts.length < 2) return;
      NB.el("path", { d: NB.line(pts.map((x) => [x, dens(x, mu)]), X, Y) + `L${X(pts[pts.length - 1])},${Y(0)}L${X(pts[0])},${Y(0)}Z`, fill: color, "fill-opacity": 0.3 }, svg);
    };
    area(cut, 2.5, 0, NB.c.blue);
    area(-1.5, cut, st.th, NB.c.orange);
    NB.el("path", { d: NB.line(xs.map((x) => [x, dens(x, 0)]), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2 }, svg);
    NB.el("path", { d: NB.line(xs.map((x) => [x, dens(x, st.th)]), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg);
    if (cut < 2.5) {
      NB.el("line", { x1: X(cut), x2: X(cut), y1: Y(0), y2: m.t, stroke: NB.c.fg, "stroke-width": 1.4, "stroke-dasharray": "4 3" }, svg);
      NB.el("text", { x: X(cut) + 5, y: m.t + 10, class: "nb-tick" }, svg, "reject H0 →");
    }
    const beta = Phi((cut - st.th) / se);
    read.innerHTML = `cutoff x̄ > <b>${NB.fmt(cut, 3)}</b> · α = <b>${NB.fmt(st.al, 2)}</b> · β = <b>${NB.fmt(beta, 3)}</b> · power 1 - β = <b>${NB.fmt(1 - beta, 3)}</b>`;
  }
  draw();
});

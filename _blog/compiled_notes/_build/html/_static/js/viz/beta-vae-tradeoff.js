/* beta-vae-tradeoff: the beta-VAE loss on a toy where everything has a closed form. Data x has 6 independent
 * Gaussian factors with variances lam_i; encoder z_i | x ~ N(a_i x_i, s_i^2), decoder xhat_i = b_i z_i with
 * Gaussian noise sig2, prior N(0, 1). Minimizing  E[(x - xhat)^2] / (2 sig2) + beta * KL  per dimension gives
 *   reconstruction error  D_i = min(lam_i, beta * sig2)
 *   compactness KL        R_i = max(0, 0.5 * ln(lam_i / (beta * sig2)))   (nats)
 * i.e. a latent dimension is switched off (KL = 0, its factor not reconstructed at all) once beta * sig2 >= lam_i.
 * Checked against scipy.optimize.minimize of the per-dimension loss over (a, b, log s^2) for several lam, beta.
 * Colors follow the note: reconstruction blue, compactness prior (KL) orange, beta teal (green token).
 * params: {beta: 1} */
NB.register("beta-vae-tradeoff", function (el, P) {
  const lam = [4, 2.5, 1.5, 0.8, 0.4, 0.15], sig2 = 0.25, K = lam.length;
  const st = { lb: Math.log10(P.beta || 1) };
  const D = (b) => lam.map((l) => Math.min(l, b * sig2));
  const R = (b) => lam.map((l) => Math.max(0, 0.5 * Math.log(l / (b * sig2))));
  const sum = (a) => a.reduce((s, v) => s + v, 0);

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "β (weight on the KL term)", min: -1, max: 1.4, step: 0.01, value: st.lb, format: (v) => NB.fmt(Math.pow(10, v), 2) }, (v) => { st.lb = v; draw(); });

  const W = 660, H = 330;
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["reconstruction error per factor", NB.c.blue], ["factor variance λᵢ (error if the latent is off)", NB.c.muted, true],
    ["compactness KL per latent", NB.c.orange], ["β σ² (water level)", NB.c.green, true]]);
  const read = NB.readout(el);

  // left: per-dimension bars; right: the rate-distortion curve traced by beta
  const L = { l: 46, r: 372, t1: 14, b1: 128, t2: 160, b2: 290 };
  const Xb = NB.scale(0, K, L.l, L.r);
  const Yk = NB.scale(0, 2.6, L.b1, L.t1), Yd = NB.scale(0, 4.2, L.b2, L.t2);
  const Rr = { l: 440, r: 646, t: 14, b: 290 };
  const betas = NB.range(-1, 1.4, 241).map((v) => Math.pow(10, v));
  const curve = betas.map((b) => [sum(R(b)), sum(D(b))]);
  const Xr = NB.scale(0, 11, Rr.l, Rr.r), Yr = NB.scale(0, 9.6, Rr.b, Rr.t);

  function draw() {
    NB.clear(svg);
    const beta = Math.pow(10, st.lb), d = D(beta), r = R(beta), lvl = beta * sig2;
    // KL bars (top left)
    NB.axes(svg, Xb, Yk, { grid: true, yticks: 3, xticks: 1, xfmt: () => "", ylabel: "KL (nats)" });
    // reconstruction bars (bottom left)
    NB.axes(svg, Xb, Yd, { grid: true, yticks: 4, xticks: 1, xfmt: () => "", ylabel: "error" });
    const bw = Xb(1) - Xb(0);
    for (let i = 0; i < K; i++) {
      const x0 = Xb(i) + bw * 0.18, w = bw * 0.64, on = r[i] > 1e-9;
      NB.el("rect", { x: x0, y: Yk(r[i]), width: w, height: Yk(0) - Yk(r[i]), fill: NB.c.orange, "fill-opacity": 0.85 }, svg);
      NB.el("rect", { x: x0, y: Yd(lam[i]), width: w, height: Yd(0) - Yd(lam[i]), fill: "none", stroke: NB.c.muted, "stroke-dasharray": "3 3" }, svg);
      const rb = NB.el("rect", { x: x0, y: Yd(d[i]), width: w, height: Yd(0) - Yd(d[i]), fill: NB.c.blue, "fill-opacity": on ? 0.8 : 0.35 }, svg);
      NB.el("title", {}, rb, `factor ${i + 1}: variance ${lam[i]}, error ${NB.fmt(d[i], 3)}, KL ${NB.fmt(r[i], 3)} nats`);
      NB.el("text", { x: Xb(i + 0.5), y: L.b2 + 16, class: "nb-tick", "text-anchor": "middle" }, svg, `z${i + 1}`);
      NB.el("text", { x: Xb(i + 0.5), y: L.b1 + 14, class: "nb-tick", "text-anchor": "middle", fill: on ? NB.c.fg : NB.c.muted }, svg, on ? "on" : "off");
    }
    if (lvl <= 4.2) NB.el("line", { x1: L.l, x2: L.r, y1: Yd(lvl), y2: Yd(lvl), stroke: NB.c.green, "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
    NB.el("text", { x: (L.l + L.r) / 2, y: L.b2 + 34, class: "nb-label", "text-anchor": "middle" }, svg, "latent dimension (one per factor)");
    // trade-off curve (right)
    NB.axes(svg, Xr, Yr, { xticks: 4, yticks: 5, xlabel: "total KL (nats)", ylabel: "total reconstruction error" });
    NB.el("path", { d: NB.line(curve, Xr, Yr), fill: "none", stroke: NB.c.muted, "stroke-width": 2 }, svg);
    const tr = sum(r), td = sum(d);
    NB.el("circle", { cx: Xr(tr), cy: Yr(td), r: 6, fill: NB.c.green }, svg);
    const nOn = r.filter((v) => v > 1e-9).length;
    read.innerHTML = `β = <b>${NB.fmt(beta, 2)}</b>: <b>${nOn}</b> of ${K} latent dimensions in use. ` +
      `<span style="color:${NB.c.blue}">reconstruction error</span> <b>${NB.fmt(td, 3)}</b> (of ${NB.fmt(sum(lam), 3)} total variance), ` +
      `<span style="color:${NB.c.orange}">compactness KL</span> <b>${NB.fmt(tr, 3)}</b> nats.`;
  }
  draw();
});

/* population-vector: N neurons with rectified cosine tuning f_a(s) = r_max [cos(s - s_a)]_+ and preferred
 * directions s_a evenly spaced around the circle (N = 4 is the cricket cercal system). Each trial the rates are
 * r_a = max(0, f_a(s) + sigma * noise). The population vector is sum_a (r_a / r_max) c_a, with c_a the unit
 * vector at s_a; its angle is the decoded direction. Left: each neuron's vector scaled by its rate, the true
 * direction and the population vector. Right: the rates against preferred direction, with the tuning curve.
 * The readout also gives the mean absolute error over 2000 noisy trials. numpy check (20000 trials, s = 60 deg,
 * sigma = 10 Hz): N = 3, 4, 8, 16, 32 give 27.0, 12.4, 8.2, 5.6, 3.9 deg. Noise-free decoding is exact for N >= 4;
 * for N = 3 only one or two neurons are active and the error reaches 30 deg (e.g. s = 30).
 * params: {dir: 60, noise: 10, n: 8} (dir in degrees, noise in Hz) */
NB.register("population-vector", function (el, P) {
  const st = { dir: P.dir ?? 60, noise: P.noise ?? 10, n: P.n ?? 8, trial: 1 };
  const RMAX = 50, rad = Math.PI / 180;
  const pref = (n) => Array.from({ length: n }, (_, a) => (360 * a) / n);
  const tune = (s, sa) => RMAX * Math.max(0, Math.cos((s - sa) * rad));
  const wrap = (d) => ((((d + 180) % 360) + 360) % 360) - 180;
  function decode(rates, prefs) {
    let x = 0, y = 0;
    rates.forEach((r, a) => { x += (r / RMAX) * Math.cos(prefs[a] * rad); y += (r / RMAX) * Math.sin(prefs[a] * rad); });
    return { x, y, ang: Math.atan2(y, x) / rad };
  }
  const trialRates = (u, prefs) => prefs.map((sa) => Math.max(0, tune(st.dir, sa) + st.noise * u.normal()));

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "true direction s (°)", min: 0, max: 359, step: 1, value: st.dir, format: (v) => v.toFixed(0) + "°" }, (v) => { st.dir = v; draw(); });
  NB.slider(ctl, { label: "noise σ (Hz)", min: 0, max: 30, step: 1, value: st.noise, format: (v) => v.toFixed(0) }, (v) => { st.noise = v; draw(); });
  NB.slider(ctl, { label: "neurons N", min: 3, max: 32, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; draw(); });
  NB.button(ctl, "new noisy trial", () => { st.trial++; draw(); });
  const row = NB.el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "10px" } }, el);
  const s1 = NB.svg(row, 320, 320), s2 = NB.svg(row, 340, 320);
  NB.legend(el, [["each neuron: rate × preferred direction", NB.c.blue], ["population vector direction (decoded)", NB.c.orange], ["true direction", NB.c.fg, true], ["tuning curve f(s)", NB.c.green]]);
  const read = NB.readout(el);

  function arrow(g, x1, y1, x2, y2, color, w, dash) {
    NB.el("line", { x1, y1, x2, y2, stroke: color, "stroke-width": w, "stroke-dasharray": dash, "stroke-linecap": "round" }, g);
    const L = Math.hypot(x2 - x1, y2 - y1);
    if (L < 4) return;
    const ux = (x2 - x1) / L, uy = (y2 - y1) / L, hs = 4 + w * 2;
    NB.el("polygon", { points: `${x2},${y2} ${x2 - hs * ux + hs * 0.5 * uy},${y2 - hs * uy - hs * 0.5 * ux} ${x2 - hs * ux - hs * 0.5 * uy},${y2 - hs * uy + hs * 0.5 * ux}`, fill: color }, g);
  }

  function draw() {
    NB.clear(s1); NB.clear(s2);
    const prefs = pref(st.n), u = NB.rng((st.trial * 2654435761 + st.n * 40503) >>> 0);
    for (let k = 0; k < 10; k++) u();   // xorshift's first outputs from nearby seeds are correlated
    const rates = trialRates(u, prefs), pv = decode(rates, prefs), err = wrap(pv.ang - st.dir);
    // left: vectors in the plane
    const cx = 160, cy = 160, R = 120;
    NB.el("circle", { cx, cy, r: R, fill: "none", class: "nb-grid" }, s1);
    NB.el("line", { x1: cx - R - 8, x2: cx + R + 8, y1: cy, y2: cy, class: "nb-grid" }, s1);
    NB.el("line", { x1: cx, x2: cx, y1: cy - R - 8, y2: cy + R + 8, class: "nb-grid" }, s1);
    prefs.forEach((sa, a) => {
      const c = Math.cos(sa * rad), s = Math.sin(sa * rad);
      NB.el("circle", { cx: cx + R * c, cy: cy - R * s, r: 2.5, fill: NB.c.muted }, s1);
      const L = Math.min(1.3, rates[a] / RMAX) * R;
      if (L > 0.5) arrow(s1, cx, cy, cx + L * c, cy - L * s, NB.c.blue, 1.6);
    });
    const tc = Math.cos(st.dir * rad), ts = Math.sin(st.dir * rad);
    arrow(s1, cx, cy, cx + (R + 14) * tc, cy - (R + 14) * ts, NB.c.fg, 1.6, "5 4");
    const pc = Math.cos(pv.ang * rad), ps = Math.sin(pv.ang * rad);
    if (Math.hypot(pv.x, pv.y) > 1e-9) arrow(s1, cx, cy, cx + R * 0.92 * pc, cy - R * 0.92 * ps, NB.c.orange, 3);
    NB.el("text", { x: 4, y: 314, class: "nb-tick" }, s1, "arrow length = rate / r_max");
    // right: rates vs preferred direction, centred on the true direction
    const m = { l: 44, r: 10, t: 12, b: 40 };
    const X = NB.scale(-180, 180, m.l, 340 - m.r), Y = NB.scale(0, 80, 320 - m.b, m.t);
    NB.axes(s2, X, Y, { xlabel: "preferred direction − s (°)", ylabel: "rate (Hz)", xticks: 4, yticks: 4 });
    NB.el("path", { d: NB.line(NB.range(-180, 180, 181).map((d) => [d, tune(d, 0)]), X, Y), fill: "none", stroke: NB.c.green, "stroke-width": 2 }, s2);
    prefs.forEach((sa, a) => {
      const d = wrap(sa - st.dir);
      NB.el("line", { x1: X(d), x2: X(d), y1: Y(0), y2: Y(Math.min(80, rates[a])), stroke: NB.c.blue, "stroke-width": 1.4, "stroke-opacity": 0.5 }, s2);
      NB.el("circle", { cx: X(d), cy: Y(Math.min(80, rates[a])), r: 3.5, fill: NB.c.blue }, s2);
    });
    // average error over many trials (fixed seed, so it doesn't jitter)
    const u2 = NB.rng(777 + st.n);
    for (let k = 0; k < 10; k++) u2();
    let tot = 0;
    for (let k = 0; k < 2000; k++) tot += Math.abs(wrap(decode(trialRates(u2, prefs), prefs).ang - st.dir));
    read.innerHTML = `This trial: decoded <b>${NB.fmt(((pv.ang % 360) + 360) % 360, 0)}°</b> vs true ${st.dir}°, error <b>${NB.fmt(Math.abs(err), 1)}°</b>. ` +
      `Average error over 2000 trials with ${st.n} neurons and σ = ${st.noise} Hz: <b>${NB.fmt(tot / 2000, 1)}°</b>.`;
  }
  draw();
});

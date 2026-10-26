/* hodgkin-huxley: the standard squid-axon model (Hodgkin & Huxley 1952, modern sign convention, rest near -65 mV)
 *   -C dV/dt = gL (V - EL) + gK n^4 (V - EK) + gNa m^3 h (V - ENa) - Ie
 * C = 1 uF/cm^2, gNa = 120, gK = 36, gL = 0.3 mS/cm^2, ENa = 50, EK = -77, EL = -54.387 mV, standard alpha/beta
 * rate functions. RK4 with dt = 0.01 ms from the resting state. One current pulse at t = 5 ms (amplitude, duration),
 * and optionally a second identical pulse some ms later to probe the refractory period. Top: V(t) and Ie(t).
 * Bottom: the three membrane currents (leak blue, K orange, Na green, as in the note's equation) or the gates.
 * Checked against scipy solve_ivp (rtol 1e-9): threshold for a 1 ms pulse 6.92 uA/cm^2 (0.5 ms: 13.3, 2 ms: 3.86);
 * Ie = 10 for 1 ms gives a spike crossing 0 mV at 7.28 ms, peak 39.1 mV at 7.52 ms, undershoot to -76.2 mV;
 * a second pulse 14 ms later fails, 15 ms later fires. The widget's bisection gives 6.9 for 1 ms.
 * params: {I: 10, dur: 1, gap: 12, view: "currents" | "gates"} */
NB.register("hodgkin-huxley", function (el, P) {
  const st = { I: P.I ?? 10, dur: P.dur ?? 1, gap: P.gap ?? 12, view: P.view || "currents" };
  const C = 1, gNa = 120, gK = 36, gL = 0.3, ENa = 50, EK = -77, EL = -54.387, T = 50, dt = 0.01, T0 = 5;
  const lin = (x) => (Math.abs(x) < 1e-6 ? 1 + x / 2 : x / (1 - Math.exp(-x)));   // x / (1 - e^-x), finite at 0
  const an = (V) => 0.1 * lin((V + 55) / 10), bn = (V) => 0.125 * Math.exp(-(V + 65) / 80);
  const am = (V) => lin((V + 40) / 10), bm = (V) => 4 * Math.exp(-(V + 65) / 18);
  const ah = (V) => 0.07 * Math.exp(-(V + 65) / 20), bh = (V) => 1 / (1 + Math.exp(-(V + 35) / 10));
  const cur = (V, m, h, n) => [gL * (V - EL), gK * n ** 4 * (V - EK), gNa * m ** 3 * h * (V - ENa)];
  const deriv = (y, I) => {
    const [V, m, h, n] = y, c = cur(V, m, h, n);
    return [(I - c[0] - c[1] - c[2]) / C, am(V) * (1 - m) - bm(V) * m, ah(V) * (1 - h) - bh(V) * h, an(V) * (1 - n) - bn(V) * n];
  };
  const step = (y, I) => {
    const k1 = deriv(y, I), k2 = deriv(y.map((v, i) => v + 0.5 * dt * k1[i]), I);
    const k3 = deriv(y.map((v, i) => v + 0.5 * dt * k2[i]), I), k4 = deriv(y.map((v, i) => v + dt * k3[i]), I);
    return y.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  };
  // resting state: gates at steady state for -65 mV, then let it settle with no input
  let rest = [-65, am(-65) / (am(-65) + bm(-65)), ah(-65) / (ah(-65) + bh(-65)), an(-65) / (an(-65) + bn(-65))];
  for (let i = 0; i < 10000; i++) rest = step(rest, 0);
  const pulses = (I, dur, gap) => [[T0, T0 + dur], ...(gap > 0 ? [[T0 + gap, T0 + gap + dur]] : [])];
  const Ie = (t, I, pl) => (pl.some(([a, b]) => t >= a - 1e-9 && t < b - 1e-9) ? I : 0);
  function simulate(I, dur, gap, keep) {
    const pl = pulses(I, dur, gap), out = [], spikes = [];
    let y = rest.slice();
    const N = Math.round(T / dt);
    for (let i = 0; i <= N; i++) {
      const t = i * dt;
      if (keep && i % 5 === 0) out.push([t, ...y, Ie(t, I, pl)]);
      const y2 = step(y, Ie(t, I, pl));
      if (y[0] < 0 && y2[0] >= 0) spikes.push(t + dt);
      y = y2;
    }
    return { out, spikes, pl };
  }
  const thrCache = {};
  function threshold(dur) {        // smallest single-pulse amplitude that gives a spike, by bisection
    if (thrCache[dur] !== undefined) return thrCache[dur];
    let lo = 0, hi = 200;
    for (let k = 0; k < 14; k++) { const mid = (lo + hi) / 2; if (simulate(mid, dur, 0, false).spikes.length) hi = mid; else lo = mid; }
    return (thrCache[dur] = hi);
  }

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "pulse current Iₑ (µA/cm²)", min: 0, max: 40, step: 0.5, value: st.I, format: (v) => v.toFixed(1) }, (v) => { st.I = v; draw(); });
  NB.slider(ctl, { label: "pulse duration (ms)", min: 0.2, max: 5, step: 0.1, value: st.dur, format: (v) => v.toFixed(1) }, (v) => { st.dur = v; draw(); });
  NB.slider(ctl, { label: "second pulse after (ms)", min: 0, max: 30, step: 1, value: st.gap, format: (v) => (v ? v.toFixed(0) : "off") }, (v) => { st.gap = v; draw(); });
  NB.select(ctl, "bottom panel", [["currents", "membrane currents"], ["gates", "gating variables m, h, n"]], st.view, (v) => { st.view = v; draw(); });
  const W = 640, H1 = 270, H2 = 200, m = { l: 54, r: 12, t: 12, b: 36 };
  const s1 = NB.svg(el, W, H1), s2 = NB.svg(el, W, H2);
  const leg = NB.el("div", {}, el);
  const read = NB.readout(el);
  const X = NB.scale(0, T, m.l, W - m.r);

  function draw() {
    NB.clear(s1); NB.clear(s2); NB.clear(leg);
    const { out, spikes, pl } = simulate(st.I, st.dur, st.gap, true);
    // top: V(t), with the injected current as a strip underneath
    const Y = NB.scale(-90, 50, H1 - m.b - 56, m.t);
    NB.axes(s1, NB.scale(0, T, m.l, W - m.r), Y, { ylabel: "V (mV)", xticks: 5, yticks: 7, xfmt: () => "" });
    NB.el("path", { d: NB.line(out.map((r) => [r[0], r[1]]), X, Y), fill: "none", stroke: NB.c.fg, "stroke-width": 1.8 }, s1);
    const yb = H1 - m.b, ytop = yb - 36, Imax = 40;
    NB.el("line", { x1: X(0), x2: X(T), y1: yb, y2: yb, class: "nb-axis" }, s1);
    NB.ticks(0, T, 5).forEach((t) => NB.el("text", { x: X(t), y: yb + 16, class: "nb-tick", "text-anchor": "middle" }, s1, NB.fmt(t)));
    NB.el("text", { x: (m.l + W - m.r) / 2, y: yb + 33, class: "nb-label", "text-anchor": "middle" }, s1, "time (ms)");
    NB.el("text", { x: m.l - 6, y: yb - 12, class: "nb-tick", "text-anchor": "end", style: { fill: NB.c.pink } }, s1, "Iₑ");
    NB.el("path", { d: NB.line(out.map((r) => [r[0], r[5]]), X, NB.scale(0, Imax, yb, ytop)), fill: "none", stroke: NB.c.pink, "stroke-width": 2 }, s1);
    // bottom
    let items;
    if (st.view === "currents") {
      const cs = out.map((r) => [r[0], ...cur(r[1], r[2], r[3], r[4])]);
      let lo = Math.min(-50, ...cs.map((r) => r[3])), hi = Math.max(50, ...cs.map((r) => r[2]));
      lo = Math.floor(lo / 100) * 100; hi = Math.ceil(hi / 100) * 100;
      const Yc = NB.scale(lo, hi, H2 - m.b, m.t);
      NB.axes(s2, X, Yc, { xlabel: "time (ms)", ylabel: "current (µA/cm²)", xticks: 5, yticks: 5 });
      [[1, NB.c.blue], [2, NB.c.orange], [3, NB.c.green]].forEach(([k, c]) =>
        NB.el("path", { d: NB.line(cs.map((r) => [r[0], r[k]]), X, Yc), fill: "none", stroke: c, "stroke-width": 1.9 }, s2));
      items = [["leak current", NB.c.blue], ["potassium current", NB.c.orange], ["sodium current (negative = inward)", NB.c.green], ["injected current Iₑ", NB.c.pink]];
    } else {
      const Yg = NB.scale(0, 1, H2 - m.b, m.t);
      NB.axes(s2, X, Yg, { xlabel: "time (ms)", ylabel: "gate", xticks: 5, yticks: 4 });
      [[2, NB.c.green, null], [3, NB.c.green, "5 4"], [4, NB.c.orange, null]].forEach(([k, c, da]) =>
        NB.el("path", { d: NB.line(out.map((r) => [r[0], r[k]]), X, Yg), fill: "none", stroke: c, "stroke-width": 1.9, "stroke-dasharray": da }, s2));
      items = [["m (Na activation)", NB.c.green], ["h (Na inactivation)", NB.c.green, true], ["n (K activation)", NB.c.orange], ["injected current Iₑ", NB.c.pink]];
    }
    NB.legend(leg, items);
    const thr = threshold(st.dur);
    const fired = pl.map(([a, b], i) => spikes.some((s) => s >= a && s < (pl[i + 1] ? pl[i + 1][0] : T)));
    let txt = `A single ${NB.fmt(st.dur, 1)} ms pulse needs at least <b>${NB.fmt(thr, 1)} µA/cm²</b> to fire. `;
    txt += `Pulse 1 (${NB.fmt(st.I, 1)}): <b>${fired[0] ? "spike" : "no spike"}</b>`;
    if (pl.length > 1) {
      txt += `; pulse 2, ${st.gap} ms later: <b>${fired[1] ? "spike" : "no spike"}</b>`;
      if (fired[0] && !fired[1] && st.I >= thr) txt += " (refractory: h is still low and n still high from the first spike)";
    }
    read.innerHTML = txt + ".";
  }
  draw();
});

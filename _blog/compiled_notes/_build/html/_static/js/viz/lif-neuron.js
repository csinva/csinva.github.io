/* lif-neuron: leaky integrate-and-fire neuron, τ dV/dt = −(V − E_L) + R I_e. When V reaches V_th a spike is
 * fired and V is reset to V_reset (= E_L here) for a refractory period. Left: V(t) for a constant input.
 * Right: firing rate vs. input (f-I curve), with the closed form 1 / (t_ref + τ ln(RI / (RI − (V_th − E_L)))).
 * params: {I: 20, tau: 20} (R I_e in mV above rest, τ in ms) */
NB.register("lif-neuron", function (el, P) {
  const st = { I: P.I || 20, tau: P.tau || 20 };
  const EL = -65, VTH = -50, TREF = 2, TMAX = 200, dt = 0.05;
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "input R·I (mV)", min: 0, max: 40, step: 0.5, value: st.I, format: (v) => v.toFixed(1) }, (v) => { st.I = v; draw(); });
  NB.slider(ctl, { label: "membrane τ (ms)", min: 5, max: 50, step: 1, value: st.tau, format: (v) => v.toFixed(0) }, (v) => { st.tau = v; draw(); });
  const row = NB.el("div", { class: "nb-row", style: { display: "grid", gridTemplateColumns: "minmax(0,1.7fr) minmax(0,1fr)", gap: "14px" } }, el);
  const s1 = NB.svg(row, 440, 300), s2 = NB.svg(row, 280, 300);
  NB.legend(el, [["membrane potential V", NB.c.blue], ["threshold V_th", NB.c.orange, true], ["f-I curve", NB.c.green]]);
  const read = NB.readout(el);
  const m = { l: 50, r: 10, t: 12, b: 40 };
  const rate = (I, tau) => (I <= VTH - EL ? 0 : 1000 / (TREF + tau * Math.log(I / (I - (VTH - EL)))));
  function draw() {
    NB.clear(s1); NB.clear(s2);
    const X = NB.scale(0, TMAX, m.l, 440 - m.r), Y = NB.scale(-70, 0, 300 - m.b, m.t);
    NB.axes(s1, X, Y, { xlabel: "time (ms)", ylabel: "V (mV)", xticks: 4, yticks: 7 });
    NB.el("line", { x1: X(0), x2: X(TMAX), y1: Y(VTH), y2: Y(VTH), stroke: NB.c.orange, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, s1);
    let V = EL, ref = 0, n = 0, d = `M${X(0).toFixed(1)},${Y(V).toFixed(1)}`;
    for (let i = 1; i * dt <= TMAX; i++) {
      const t = i * dt;
      if (ref > 0) { ref -= dt; V = EL; }
      else {
        V += (dt / st.tau) * (-(V - EL) + st.I);
        if (V >= VTH) {   // spike: draw a stroke up to 0 mV, then reset
          n++; d += `L${X(t).toFixed(1)},${Y(VTH).toFixed(1)}L${X(t).toFixed(1)},${Y(-2).toFixed(1)}L${X(t).toFixed(1)},${Y(EL).toFixed(1)}`;
          V = EL; ref = TREF; continue;
        }
      }
      if (i % 4 === 0) d += `L${X(t).toFixed(1)},${Y(V).toFixed(1)}`;
    }
    NB.el("path", { d, fill: "none", stroke: NB.c.blue, "stroke-width": 1.8 }, s1);
    // f-I curve
    const XI = NB.scale(0, 40, m.l, 280 - m.r);
    const YI = NB.scale(0, 250, 300 - m.b, m.t);
    NB.axes(s2, XI, YI, { xlabel: "input R·I (mV)", ylabel: "rate (Hz)", xticks: 4, yticks: 4 });
    const Is = NB.range(0, 40, 401);
    NB.el("path", { d: NB.line(Is.map((I) => [I, Math.min(250, rate(I, st.tau))]), XI, YI), fill: "none", stroke: NB.c.green, "stroke-width": 2.2 }, s2);
    const r = rate(st.I, st.tau);
    NB.el("circle", { cx: XI(st.I), cy: YI(Math.min(250, r)), r: 5.5, fill: NB.c.green }, s2);
    NB.el("line", { x1: XI(VTH - EL), x2: XI(VTH - EL), y1: YI(0), y2: YI(250), stroke: NB.c.orange, "stroke-dasharray": "3 3", "stroke-width": 1 }, s2);
    read.innerHTML = `${n} spikes in ${TMAX} ms; predicted rate <b>${NB.fmt(r, 1)} Hz</b>` +
      (st.I <= VTH - EL ? ` (input below V_th − E_L = ${VTH - EL} mV: V settles at ${NB.fmt(EL + st.I, 1)} mV and never fires)` : "");
  }
  draw();
});

/* bayes-update: estimating a coin's heads probability θ. Beta(a, b) prior x binomial likelihood -> Beta posterior.
 * Colors match the chapter's equation: posterior green, likelihood blue, prior orange.
 * params: {a: 2, b: 2, heads: 7, flips: 10} */
NB.register("bayes-update", function (el, P) {
  const st = { a: P.a || 2, b: P.b || 2, h: P.heads ?? 7, n: P.flips || 10 };
  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "prior a (pseudo-heads)", min: 0.5, max: 20, step: 0.5, value: st.a }, (v) => { st.a = v; draw(); });
  NB.slider(ctl, { label: "prior b (pseudo-tails)", min: 0.5, max: 20, step: 0.5, value: st.b }, (v) => { st.b = v; draw(); });
  const flips = NB.slider(ctl, { label: "flips", min: 1, max: 100, step: 1, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; st.h = Math.min(st.h, v); heads.input.max = v; heads.set(st.h); draw(); });
  const heads = NB.slider(ctl, { label: "heads", min: 0, max: st.n, step: 1, value: st.h, format: (v) => v.toFixed(0) }, (v) => { st.h = v; draw(); });
  const W = 640, H = 290, m = { l: 44, r: 10, t: 10, b: 34 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["prior p(θ)", NB.c.orange], ["likelihood p(x | θ), rescaled", NB.c.blue], ["posterior p(θ | x)", NB.c.green]]);
  const read = NB.readout(el);
  const lgamma = (z) => { // Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  };
  const beta = (t, a, b) => (t <= 0 || t >= 1 ? 0 : Math.exp((a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) - (lgamma(a) + lgamma(b) - lgamma(a + b))));
  function draw() {
    NB.clear(svg);
    const ts = NB.range(0.001, 0.999, 300);
    const prior = ts.map((t) => beta(t, st.a, st.b)), post = ts.map((t) => beta(t, st.a + st.h, st.b + st.n - st.h));
    const likeRaw = ts.map((t) => Math.exp(st.h * Math.log(t) + (st.n - st.h) * Math.log(1 - t)));
    const lmax = Math.max(...likeRaw), pmax = Math.max(...post, ...prior.filter(isFinite));
    const like = likeRaw.map((l) => (l / lmax) * pmax * 0.8);
    const X = NB.scale(0, 1, m.l, W - m.r), Y = NB.scale(0, Math.min(pmax * 1.1, 40), H - m.b, m.t);
    NB.axes(svg, X, Y, { xlabel: "θ = probability of heads", ylabel: "density", xticks: 5, yticks: 4 });
    const cap = (arr) => ts.map((t, i) => [t, Math.min(arr[i], Y.domain[1])]);
    NB.el("path", { d: NB.line(cap(prior), X, Y), fill: "none", stroke: NB.c.orange, "stroke-width": 2 }, svg);
    NB.el("path", { d: NB.line(cap(like), X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    NB.el("path", { d: NB.line(cap(post), X, Y), fill: "none", stroke: NB.c.green, "stroke-width": 2.6 }, svg);
    const mle = st.h / st.n, map = (st.a + st.h - 1) / (st.a + st.b + st.n - 2), mean = (st.a + st.h) / (st.a + st.b + st.n);
    read.innerHTML = `MLE <b>${NB.fmt(mle, 3)}</b> · MAP <b>${NB.fmt(map, 3)}</b> · posterior mean (Bayes estimate) <b>${NB.fmt(mean, 3)}</b>. More flips pull all three together; a stronger prior pulls MAP and the mean toward ${NB.fmt(st.a / (st.a + st.b), 2)}.`;
  }
  draw();
});

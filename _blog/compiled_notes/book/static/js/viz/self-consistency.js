/* self-consistency: majority vote over k sampled reasoning paths (wang et al. 2022), in an idealized model.
 * Each sample independently gives the correct answer with probability p, otherwise one of m = 4 wrong answers;
 * a share s of the wrong mass goes to one wrong answer, the rest is split evenly over the other three.
 * Accuracy of the vote is exact: sum over the correct answer's count c of P(c) * P(every wrong count <= c),
 * ties broken uniformly at random (a DP over the wrong answers). Checked against brute-force multinomial
 * enumeration and a 200k-trial simulation in numpy.
 * params: {p: 0.4, share: 0.4, k: 9} */
NB.register("self-consistency", function (el, P) {
  const st = { p: P.p === undefined ? 0.4 : P.p, s: P.share === undefined ? 0.4 : P.share, k: P.k || 9 };
  const m = 4, kmax = 40;
  const answers = ["18", "16", "20", "24", "9"];                     // index 0 is the correct answer
  const cols = [NB.c.green, NB.c.orange, NB.c.purple, NB.c.pink, NB.c.yellow];
  const probs = () => { const w = 1 - st.p; return [st.p, w * st.s, ...Array(m - 1).fill((w * (1 - st.s)) / (m - 1))]; };
  const fact = [1]; for (let i = 1; i <= kmax; i++) fact[i] = fact[i - 1] * i;

  // P(majority vote over k samples is correct), ties split evenly
  function voteAcc(k, pr) {
    let tot = 0;
    for (let c = 1; c <= k; c++) {
      const r = k - c;
      let dp = [Array(m + 1).fill(0)]; dp[0][0] = 1;              // dp[n][t]: n wrong votes used, t wrong answers tied with c
      for (let j = 1; j <= m; j++) {
        const nd = Array.from({ length: r + 1 }, () => Array(m + 1).fill(0)), q = pr[j];
        dp.forEach((row, n) => row.forEach((v, t) => {
          if (!v) return;
          for (let a = 0; a <= Math.min(c, r - n); a++) nd[n + a][t + (a === c ? 1 : 0)] += (v * Math.pow(q, a)) / fact[a];
        }));
        dp = nd;
      }
      const g = (dp[r] || []).reduce((s, v, t) => s + v / (t + 1), 0);
      tot += (fact[k] * Math.pow(pr[0], c)) / fact[c] * g;
    }
    return Math.min(1, tot);
  }

  const ctl = NB.controls(el);
  NB.slider(ctl, { label: "p = P(one sample is right)", min: 0.05, max: 0.95, step: 0.01, value: st.p }, (v) => { st.p = v; resample(false); });
  NB.slider(ctl, { label: "wrong answers: share of the top one", min: 0.25, max: 1, step: 0.01, value: st.s }, (v) => { st.s = v; resample(false); });
  NB.slider(ctl, { label: "samples k", min: 1, max: kmax, step: 1, value: st.k, format: (v) => v.toFixed(0) }, (v) => { st.k = v; resample(false); });
  NB.button(ctl, "sample a new vote", () => resample(true));

  const W = 640, H = 270, m2 = { l: 48, r: 14, t: 12, b: 40 };
  const svg = NB.svg(el, W, H);
  const H2 = 112, svg2 = NB.svg(el, W, H2);
  NB.legend(el, [["majority vote of k samples", NB.c.blue], ["one sample (p)", NB.c.green, true], ["most common wrong answer", NB.c.orange, true]]);
  const read = NB.readout(el);
  const X = NB.scale(1, kmax, m2.l, W - m2.r), Y = NB.scale(0, 1, H - m2.b, m2.t);
  const rng = NB.rng(P.seed || 2203);
  for (let i = 0; i < 20; i++) rng();
  let votes = [], timer = null;

  function draw() {
    NB.clear(svg);
    const pr = probs();
    NB.axes(svg, X, Y, { xlabel: "number of sampled reasoning paths k", ylabel: "accuracy", xticks: 8, yticks: 5 });
    NB.el("line", { x1: X(1), x2: X(kmax), y1: Y(st.p), y2: Y(st.p), stroke: NB.c.green, "stroke-dasharray": "6 4", "stroke-width": 1.6 }, svg);
    NB.el("line", { x1: X(1), x2: X(kmax), y1: Y(pr[1]), y2: Y(pr[1]), stroke: NB.c.orange, "stroke-dasharray": "3 4", "stroke-width": 1.4 }, svg);
    const pts = []; for (let k = 1; k <= kmax; k++) pts.push([k, voteAcc(k, pr)]);
    NB.el("path", { d: NB.line(pts, X, Y), fill: "none", stroke: NB.c.blue, "stroke-width": 2.4 }, svg);
    pts.forEach(([k, a]) => NB.el("circle", { cx: X(k), cy: Y(a), r: k === st.k ? 5.5 : 2.2, fill: k === st.k ? NB.c.blue : NB.c.blue, stroke: k === st.k ? NB.c.bg : "none", "stroke-width": 1.5 }, svg));
    return pts;
  }

  function drawVote(nShown) {
    NB.clear(svg2);
    const k = votes.length, x0 = 12, cw = Math.min(30, (W - 24) / Math.max(k, 1)), r = Math.min(12, cw / 2 - 1.5);
    NB.el("text", { x: x0, y: 16, class: "nb-label" }, svg2, `one vote with k = ${k} (correct answer: ${answers[0]})`);
    const counts = Array(m + 1).fill(0);
    votes.slice(0, nShown).forEach((a, i) => {
      counts[a]++;
      const cx = x0 + i * cw + cw / 2;
      NB.el("circle", { cx, cy: 40, r, fill: cols[a], "fill-opacity": 0.25, stroke: cols[a], "stroke-width": 1.5 }, svg2);
      if (r >= 8) NB.el("text", { x: cx, y: 44, "text-anchor": "middle", style: { fontSize: "11px", fontWeight: 600 } }, svg2, answers[a]);
    });
    // tally
    const top = Math.max(...counts), bw = (W - 24) / (m + 1);
    counts.forEach((n, a) => {
      const bx = x0 + a * bw, len = k ? ((bw - 70) * n) / k : 0;
      NB.el("text", { x: bx, y: 86, class: "nb-tick", style: { fill: cols[a], fontWeight: 600 } }, svg2, answers[a]);
      NB.el("rect", { x: bx + 22, y: 75, width: Math.max(len, 0.5), height: 14, rx: 2, fill: cols[a], "fill-opacity": n === top && nShown === k ? 0.9 : 0.35 }, svg2);
      NB.el("text", { x: bx + 26 + len, y: 86, class: "nb-tick" }, svg2, n);
    });
    if (nShown === k) {
      const winners = counts.map((n, a) => (n === top ? a : -1)).filter((a) => a >= 0);
      const txt = winners.length > 1 ? `tie between ${winners.map((a) => answers[a]).join(" and ")}: broken at random`
        : `majority answer ${answers[winners[0]]}: ${winners[0] === 0 ? "correct" : "wrong"}`;
      NB.el("text", { x: x0, y: 106, class: "nb-tick", style: { fill: winners.length === 1 && winners[0] === 0 ? NB.c.green : NB.c.fg } }, svg2, txt);
    }
  }

  function resample(animate) {
    if (timer) { clearInterval(timer); timer = null; }
    const pr = probs(), cum = pr.map((_, i) => pr.slice(0, i + 1).reduce((a, b) => a + b, 0));
    votes = Array.from({ length: st.k }, () => { const u = rng(); const i = cum.findIndex((c) => u < c); return i < 0 ? m : i; });
    const pts = draw();
    const acc = pts[st.k - 1][1], pw = pr[1];
    const lim = st.p > pw ? "the correct answer is the single most likely one, so the vote's accuracy climbs toward 1 as k grows"
      : st.p < pw ? "one wrong answer is more likely than the correct one, so the vote converges on that wrong answer and accuracy falls toward 0"
      : "the correct answer and the top wrong answer are equally likely, so the vote stays a coin flip between them";
    read.innerHTML = `One sample is right with probability <b>${NB.fmt(st.p, 2)}</b>; the majority of k = ${st.k} is right with probability <b>${NB.fmt(acc, 3)}</b>. ` +
      `The most common wrong answer has probability ${NB.fmt(pw, 2)}: ${lim}.`;
    if (!animate) { drawVote(st.k); return; }
    let i = 0; drawVote(0);
    timer = setInterval(() => { i++; drawVote(i); if (i >= st.k) { clearInterval(timer); timer = null; } }, Math.max(25, 600 / st.k));
  }
  resample(false);
});

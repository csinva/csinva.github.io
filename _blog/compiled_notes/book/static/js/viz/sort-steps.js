/* sort-steps: run one sorting algorithm on a small array and scrub through it step by step.
 * Bars are the array; the two compared positions are highlighted. The counters compare the number of
 * comparisons with n²/2 and n·log2(n).
 * params: {algo: "insertion" | "bubble" | "selection" | "merge" | "quick", input: "random" | "nearly" | "reversed", n: 32} */
NB.register("sort-steps", function (el, P) {
  const st = { algo: P.algo || "insertion", input: P.input || "random", n: P.n || 32, t: 0 };
  const ctl = NB.controls(el);
  NB.select(ctl, "algorithm", [["bubble", "bubble sort"], ["selection", "selection sort"], ["insertion", "insertion sort"],
    ["merge", "merge sort"], ["quick", "quicksort (last element as pivot)"]], st.algo, (v) => { st.algo = v; rerun(); });
  NB.select(ctl, "input", [["random", "random order"], ["nearly", "nearly sorted"], ["reversed", "reversed"]], st.input, (v) => { st.input = v; rerun(); });
  NB.slider(ctl, { label: "n", min: 8, max: 64, step: 4, value: st.n, format: (v) => v.toFixed(0) }, (v) => { st.n = v; rerun(); });
  const ctl2 = NB.controls(el);
  const play = NB.button(ctl2, "▶ play", () => toggle());
  const tS = NB.slider(ctl2, { label: "step", min: 0, max: 1, step: 1, value: 0, format: (v) => v.toFixed(0) }, (v) => { stop(); st.t = v; draw(); });
  tS.input.style.width = "16rem";
  const W = 640, H = 180, m = { l: 6, r: 6, t: 8, b: 8 };
  const svg = NB.svg(el, W, H);
  NB.legend(el, [["being compared", NB.c.orange], ["just written", NB.c.pink]]);
  const read = NB.readout(el);
  let frames = [], timer = null;

  function makeInput() {
    const n = st.n, u = NB.rng(3 + n), a = Array.from({ length: n }, (_, i) => i + 1);
    if (st.input === "reversed") return a.reverse();
    if (st.input === "nearly") { for (let k = 0; k < Math.max(1, n / 16); k++) { const i = Math.floor(u() * (n - 1)); [a[i], a[i + 1]] = [a[i + 1], a[i]]; } return a; }
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  // record a frame after every comparison or write: {a, cmp: [i, j], set: [i...], c: comparisons so far, w: writes so far}
  function record(a0) {
    const a = a0.slice(), F = [{ a: a.slice(), cmp: [], set: [], c: 0, w: 0 }];
    let c = 0, w = 0;
    const less = (i, j) => { c++; F.push({ a: a.slice(), cmp: [i, j], set: [], c, w }); return a[i] < a[j]; };
    const lessV = (x, y, i, j) => { c++; F.push({ a: a.slice(), cmp: [i, j], set: [], c, w }); return x < y; };
    const swap = (i, j) => { [a[i], a[j]] = [a[j], a[i]]; w += 2; F.push({ a: a.slice(), cmp: [], set: [i, j], c, w }); };
    const put = (i, v) => { a[i] = v; w++; F.push({ a: a.slice(), cmp: [], set: [i], c, w }); };
    const n = a.length;
    if (st.algo === "bubble") {
      for (let pass = 0; pass < n - 1; pass++) {
        let swapped = false;                                    // flag: a pass with no swaps means done
        for (let i = 0; i < n - 1 - pass; i++) if (less(i + 1, i)) { swap(i, i + 1); swapped = true; }
        if (!swapped) break;
      }
    } else if (st.algo === "selection") {
      for (let i = n - 1; i > 0; i--) {                       // move the largest to the current position
        let jmax = 0;
        for (let j = 1; j <= i; j++) if (less(jmax, j)) jmax = j;
        if (jmax !== i) swap(jmax, i);
      }
    } else if (st.algo === "insertion") {
      for (let i = 1; i < n; i++) { let j = i; while (j > 0 && less(j, j - 1)) { swap(j, j - 1); j--; } }
    } else if (st.algo === "merge") {
      const sort = (lo, hi) => {
        if (hi - lo < 2) return;
        const mid = (lo + hi) >> 1; sort(lo, mid); sort(mid, hi);
        const L = a.slice(lo, mid), R = a.slice(mid, hi);
        let i = 0, j = 0, k = lo;
        while (i < L.length && j < R.length) put(k++, lessV(R[j], L[i], mid + j, lo + i) ? R[j++] : L[i++]);
        while (i < L.length) put(k++, L[i++]);
        while (j < R.length) put(k++, R[j++]);
      };
      sort(0, n);
    } else {
      const qs = (lo, hi) => {                                // Lomuto partition around a[hi]
        if (lo >= hi) return;
        let s = lo;
        for (let j = lo; j < hi; j++) if (less(j, hi)) { if (s !== j) swap(s, j); s++; }
        if (s !== hi) swap(s, hi);
        qs(lo, s - 1); qs(s + 1, hi);
      };
      qs(0, n - 1);
    }
    return F;
  }

  function rerun() {
    stop(); frames = record(makeInput()); st.t = 0;
    tS.input.max = frames.length - 1; tS.set(0); draw();
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; } }
  function toggle() {
    if (timer) return stop();
    if (st.t >= frames.length - 1) st.t = 0;
    play.textContent = "❚❚ pause";
    const per = Math.max(1, Math.round(frames.length / 240));  // whole run takes ~8 s
    timer = setInterval(() => { st.t = Math.min(frames.length - 1, st.t + per); tS.set(st.t); draw(); if (st.t >= frames.length - 1) stop(); }, 33);
  }

  function draw() {
    NB.clear(svg);
    const f = frames[st.t], n = f.a.length, bw = (W - m.l - m.r) / n, Y = NB.scale(0, n, H - m.b, m.t);
    f.a.forEach((v, i) => {
      const col = f.cmp.includes(i) ? NB.c.orange : f.set.includes(i) ? NB.c.pink : NB.c.blue;
      NB.el("rect", { x: m.l + i * bw + 0.6, y: Y(v), width: Math.max(1, bw - 1.2), height: Y(0) - Y(v), rx: 1.5, fill: col, "fill-opacity": col === NB.c.blue ? 0.55 : 1 }, svg);
    });
    const last = frames[frames.length - 1], nn = n * (n - 1) / 2, nlog = n * Math.log2(n);
    read.innerHTML = `step ${st.t}/${frames.length - 1}: <b>${f.c}</b> comparisons, <b>${f.w}</b> writes so far. ` +
      `Whole run: ${last.c} comparisons; for n = ${n}, n(n−1)/2 = ${nn} and n·log₂n ≈ ${Math.round(nlog)}.`;
  }
  rerun();
});

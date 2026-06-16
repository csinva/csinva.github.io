Two small, self-contained web projects that live happily at https://csinva.io/quick-viz/

## 🌌 The Spectra

Interactive scatter-plot explorers ("spectra") of entire catalogs — every dot is annotated, plotted on two switchable axes, and clickable. Each page also includes a Co-Star-style personality reading generated from your 3–5 favorites.

| Page | What's plotted | Highlights |
|---|---|---|
| [`index.html`](index.html) | Landing page | Links to all spectra |
| [`taylor.html`](taylor.html) | All 252 Taylor Swift songs (every album, vault track, and single, 2006–2026) | Real lengths, lifetime + weekly Spotify streams, release dates; Spotify/YouTube play buttons; "Your Aura" readings |
| [`olivia.html`](olivia.html) | All 43 Olivia Rodrigo songs, incl. *you seem pretty sad for a girl so in love* (June 2026) | Same engine; default axis: Anger |
| [`basketball.html`](basketball.html) | 100 NBA legends, current stars, and every All-NBA 1st/2nd teamer 2000–2026 | Height, FT%, scoring, blocks, rings, clutch; "Starting Five" scouting reports |
| [`curry.html`](curry.html) | 42 curries across six traditions | Heat, creaminess, tang, soupiness, cook time; "Your Curry Order" readings |

Each page is a **single self-contained HTML file** — no build step, no dependencies, works offline (album art is base64-embedded). Open locally or serve statically.

**Data sources:** track lists, durations, and artwork from the iTunes catalog; stream counts from [kworb.net](https://kworb.net); video links resolved from YouTube; subjective dimensions (happiness, wholesomeness, clutch, curry heat…) are Claude-annotated opinion, 0–100.

## 🧠 Quick Quiz

[`quick-quiz/`](quick-quiz/) is a small quiz app for rapid-fire memorization — country flags, world capitals, Greek/Roman mythology — with datasets in [`quick-quiz/data/`](quick-quiz/data/). See [`quick-quiz/README.md`](quick-quiz/README.md) for details.

## Hosting

The repo is served as-is by GitHub Pages, so every page above is reachable directly at its path (e.g. `/taylor.html`, `/quick-quiz/`).

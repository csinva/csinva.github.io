# The notes book: how it is built and how to change it

This folder turns the markdown notes in `_notes/` into the jupyter-book at `_blog/compiled_notes/_build/html`
(linked from csinva.io as the "ML-AI Guide"). **The notes are the source of truth and are never edited by
the book code.** Everything the book adds on top of them lives here.

```
book/
  build_book.py        entry point: notes -> MyST pages + parts + start page + config -> jupyter-book
  transform.py         kramdown note -> MyST page (figures, dropdowns, math, links, mermaid, overlays)
  check_build.py       after a build: leaked markup, pages that lost content
  check_overlays.py    overlays vs. notes: drift (renamed headers, removed images), voice lint, coverage
  tools/shoot.py       screenshot any page of a build (light/dark) to look at it
  overlays/
    _landing.yml       start page text
    _parts/<cat>.yml   one per part (cs, math, stat, ml, ai, neuro, research)
    <folder>/<slug>.yml  one per note, e.g. overlays/stat/stat.yml for _notes/stat/stat.md
  static/              copied to the book's _static/
    css/book.css       the whole theme (colors are tokens; dark mode redefines them)
    js/nb-viz.js       tiny widget framework (helpers + registry)
    js/viz/<name>.js   one interactive widget per file
```

## Build

```bash
cd _blog/compiled_notes
uv run python book/build_book.py                       # full build into _build/html (what gets committed)
uv run python book/check_build.py                      # must print "0 with problems" (also syntax-checks every widget)
uv run python book/check_overlays.py [--coverage]      # must print "0 drift/voice problems"
# quick preview of a few notes in a scratch dir (doesn't touch the real build):
uv run python book/build_book.py --only stat/stat ml/classification --out-dir /tmp/bk --src-dir /tmp/bk/src
uv run python book/tools/shoot.py /tmp/bk/_build/html notes/stat/stat.html shot.png --find "bias" [--dark]
```
`--keep-src` keeps the generated MyST in `src/` so you can read exactly what jupyter-book sees.
In zsh, `--only $NOTES` passes one argument; write the notes out or use `${=NOTES}`.
The build deletes `_build/` first, so pages of removed notes disappear. A full build takes ~10 seconds.

Which notes are included: every `_notes/<folder>/*.md` with front matter and a `category` that is one
of the parts in `PARTS` (build_book.py), in the order of `_data/notes_order.yml` (the homepage order).
Notes in `misc/` and `talks/` have no category and are not in the book.

## What transform.py does to a note
- front matter title -> page title (sentence case, acronyms kept: `display_title`); subtitle -> lead line
- every `#` header moves down one level (the page title is the only h1) and starts with a capital letter
- `<img>` and `![]()` -> `{figure}` with a caption: the overlay caption if there is one, else the svg's
  `<title>`, else a meaningful alt text. Captioned figures are numbered per page ("Figure 3."; sphinx's
  numfig is off because it numbers across the whole book). Width comes from the img style. Images in
  tables stay plain. A `<p float="left">` group of images becomes a grid of side-by-side figures.
- `<details><summary>` -> `{dropdown}`; ```` ```mermaid ```` -> rendered diagram
- kramdown math -> MyST math: `$$x$$` inside text becomes `$x$`; a list item that is only a one-line
  `$$x$$` becomes inline; real multi-line equations stay display math. `\cblue{}` etc. color macros work
  (defined in build_book.py `MACROS`, colors from css tokens so they switch with the theme). Named colors
  the notes use directly (`\textcolor{teal}`, `\color{NavyBlue}`, ...) are mapped to the same tokens
  (`NAMED_COLORS` in transform.py). Pages whose only math is in widget captions get a hidden math node so
  sphinx still loads MathJax.
- links to other notes (`csinva.io/notes/<folder>/<slug>.html`) -> links between book pages
- layout-only html wrapper lines (`<p float=left>`, `<span>`, `{:toc}`) are dropped
If a note uses some new construct that renders badly, fix it in transform.py (generally, not per note)
and re-run check_build.py.

## Navigation the build generates (no per-note work needed)
- **Start page** (`overlays/_landing.yml`): lead, intro, `paths:` (reading paths: title, description, list
  of chapter ids in order), the topic map (`map_intro`, `map_caption`), then every part with its cards.
- **Part pages** (`overlays/_parts/<cat>.yml`: `intro`, `blurb`): intro, a topic map focused on the part
  (its chapters plus their closest chapters elsewhere, faded), and the chapter cards.
- **Chapters**: a breadcrumb (part and position), and a "Related chapters" footer: the overlay's
  `related:` ids first, then the most similar chapters by tf-idf, three in total.
- **Glossary** (`reference/glossary`): every `- *term* - definition` / `- **term**: definition` line in the
  notes (transform.py `DEF`), A-Z and filterable; each term gets an anchor at its definition so the glossary
  links straight to it. To get a term into the glossary, write it in that form in the note.
- **Interactive figures** (`reference/figures`): every widget, by part, linking to where it appears.
- **Linked terms**: a two-pass build. Pass 1 collects every definition; pass 2 turns the first mention of a
  term on other pages into a link to its definition with the definition shown on hover. Italic mentions
  (`*term*`) of any distinctive term are linked, and so are plain-text mentions of multi-word terms. Generic
  words are excluded (`GENERIC_TERMS` in build_book.py), as are terms defined in more than two places and
  terms defined only in research overviews. Add a word to `GENERIC_TERMS` if it gets linked where it shouldn't.
- **Chapter outline**: chapters with 6+ top-level sections get a collapsible outline under the intro: each
  section with its overlay lead (first sentence) and how many figures and interactive figures it has.
- **Cards** show reading time and number of interactive figures; every widget gets a reset button (nb-viz.js).
- The topic map's similarity comes from the note text, so it updates by itself when notes change.

## Overlays: what the book adds to a note
`overlays/<folder>/<slug>.yml` (all keys optional):
```yaml
title: Linear algebra          # only if the front-matter title needs fixing (e.g. Title Case)
subtitle: ...                  # only if the note's own subtitle is weak
hook: One sentence for the chapter card on the part page and start page.
map_label: Short name     # optional: label on the start-page topic map (long titles collide)
related: [ml/kernels, math/linear_algebra]   # optional: chapters to list first under "Related chapters"
intro: |                       # "About this chapter" box: 1-2 short paragraphs
  ...
viz: [ {name, params, caption} ]          # widgets at the top of the page (rare)
sections:                      # keyed by header text exactly as in the note (case/markup ignored)
  bias and point estimation:
    lead: |                    # 1-3 sentences shown under the header, before the note's bullets
      ...
    viz:
      - name: bias-variance    # a widget registered in static/js/viz/
        params: {degree: 3}
        caption: What the reader should look for. Inline math $x$ works.
        after: Huffman coding  # optional: place it right after the bullet containing this text (and its
                               # sub-items) instead of directly under the header
takeaways:                     # optional "Key ideas" box closing the chapter: 3-6 one-sentence points
  - The sample mean's variance is sigma^2/n, so averages settle down at rate 1/sqrt(n).
figures:                       # captions keyed by image file name (overrides the svg title)
  backprop.svg: Forward values in green, gradients in red ...
```
When a note changes, run `check_overlays.py`: it lists every key that no longer matches (a renamed
header, a removed image, a missing widget). Update the key or delete the entry. `--coverage` lists
what has no overlay yet. Only the first header with a given text gets the lead/widgets.

## Writing rules (for overlay text)
The notes are terse bullets in the author's own voice. The overlay text explains and connects them, and
must sound like the same person wrote it, not like generated text.
- **Grounded.** Only say what the note says or what is needed to read it (e.g. what an estimator is,
  when the note's bullets assume it). No new claims, numbers, papers or citations that aren't in the note.
  If you aren't sure something is right, leave it out.
- **Specific and plain.** Name the actual objects ("the sample mean", "the posterior"). Short sentences.
  Explain why a thing matters in one clause, not a paragraph. No hype, no throat-clearing, no summaries of
  what you are about to say, no rhetorical questions, no exclamation marks, no em dashes.
- **Banned** (check_overlays.py enforces): delve, crucial, pivotal, landscape, tapestry, realm, seamless,
  leverage, harness, unlock, journey, nuanced, intricate, multifaceted, paramount, underscore, showcase,
  game-changer, cutting-edge, fascinating, powerful, "it's worth noting", "in this chapter, we", "let's",
  "not only ... but also", "in summary/conclusion", "overall,".
- Intro: what the chapter covers and how its parts relate (what comes first and why), in 2 short
  paragraphs. Section leads: only for sections where a sentence of context makes the bullets easier to
  read; skip obvious ones. Captions: say what the figure shows and what to notice.
- Use the note's notation. Use `$...$` for math. Color words with `<span class="cblue">` only to match an
  equation that already uses `\cblue{}`.

## Widgets
A widget is `static/js/viz/<name>.js` calling `NB.register("<name>", (el, params) => {...})`, and is
placed with `viz:` in an overlay. Rules:
- Only illustrate something the note covers, and make the interaction teach it: a slider whose effect
  the reader can predict and check (step size -> divergence, degree -> overfitting, prior -> posterior).
  One idea per widget. A caption says what to try and what to notice.
- Use the helpers in `js/nb-viz.js`: `NB.svg` (fixed coordinates, scales to width), `NB.scale`,
  `NB.axes`, `NB.line`, `NB.slider`/`NB.select`/`NB.button`, `NB.legend`, `NB.readout`, `NB.rng` (seeded,
  so figures are stable), `NB.lstsq`/`NB.solve`. d3 v7 is available as `window.d3` if really needed.
- Colors only via `NB.c.*` / `var(--nb-c-*)` so light and dark mode both work. SVG `<text>` gets the
  foreground color unless it has its own `fill` attribute. Captions may use `<span class="cblue">..</span>`
  (only these color spans pass through; everything else in a caption is escaped) and `$..$` math. When the note colors an
  equation, use the same color for the same quantity in the widget.
- No network requests (except `_static/data/*` files the build writes), no external libraries, no
  `localStorage`. Must work at phone width (svg scales; controls wrap).
- Numerics must be right: check a widget's numbers against numpy before trusting a picture.
- `NB.rng(seed)` is xorshift32 (its `>> 17` is a signed shift if you port it to python). Its first draws are
  correlated for small nearby seeds: if you derive many seeds (one per trial), hash the seed and discard a
  few draws inside your widget. Don't change NB.rng itself, every existing widget's data depends on it.
- Place a widget with `uv run python book/tools/place_viz.py placements.yml` (keeps the overlay's formatting).
- Check every widget in both themes with tools/shoot.py.

## Theme
`static/css/book.css` holds all styling (the start page and part overviews keep both sidebars open via a hidden `.sidebars-open` marker; the top previous/next links are copied from the footer by nb-viz.js; on wide screens both sidebars tuck into 48px strips and slide out on hover/focus: "distraction-free reading" block); colors are tokens on `:root` redefined under
`html[data-theme="dark"]`. Notes' images are drawn dark-on-transparent and get inverted in dark mode
(same as the website). Fonts: Inter (text), Source Serif 4 (headings), JetBrains Mono (code).

## After changing anything
1. `uv run python book/build_book.py && uv run python book/check_build.py && uv run python book/check_overlays.py`
2. screenshot the pages you touched (light and dark) and look at them
3. commit `_build/html` together with the change (csinva.io serves the built files directly)

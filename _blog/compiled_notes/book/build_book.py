"""Build the jupyter-book version of the notes.

    cd _blog/compiled_notes && uv run python book/build_book.py [--keep-src] [--only ml/classification ...]

Steps
1. read every published note in _notes/ (front matter with a `category`), in the homepage order from
   _data/notes_order.yml
2. convert each one to a MyST page (book/transform.py), merging in its overlay from book/overlays/
3. write part landing pages, the start page, topic-map data, _toc.yml and _config.yml into src/
4. run jupyter-book, writing the site to _build/html (this path is what csinva.io links to)
5. delete src/ and build leftovers that shouldn't be committed

The notes are never modified. See book/AGENTS.md for how to change or extend the book.
"""
import argparse
import html
import json
import pathlib
import re
import shutil
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent            # _blog/compiled_notes/book
ROOT = HERE.parent                                         # _blog/compiled_notes
REPO = ROOT.parent.parent
NOTES = REPO / "_notes"
SRC = ROOT / "src"
OUT = ROOT / "_build"
sys.path.insert(0, str(HERE))
from transform import Context, display_title, note_to_myst  # noqa: E402

# book parts, in reading order: (category in front matter, folder in _notes, part title)
PARTS = [
    ("cs", "cs", "Computer science"),
    ("math", "math", "Mathematics"),
    ("stat", "stat", "Statistics"),
    ("ml", "ml", "Machine learning"),
    ("ai", "ai", "Artificial intelligence"),
    ("neuro", "neuro", "Neuroscience"),
    ("research", "research_ovws", "Research overviews"),
]
CDN_D3 = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"
CDN_MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
         "&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400&display=swap")
MACROS = {  # same TeX shorthands + color macros as the site (_layouts/notes.html), darker for light pages
    "bold": ["\\mathbf{#1}", 1], "underbar": ["\\underline{#1}", 1], "ra": "\\rightarrow", "empty": "\\emptyset",
    "cblue": ["{\\color{var(--nb-c-blue)}{#1}}", 1], "corange": ["{\\color{var(--nb-c-orange)}{#1}}", 1],
    "cgreen": ["{\\color{var(--nb-c-green)}{#1}}", 1], "cpink": ["{\\color{var(--nb-c-pink)}{#1}}", 1],
    "cpurple": ["{\\color{var(--nb-c-purple)}{#1}}", 1], "cyellow": ["{\\color{var(--nb-c-yellow)}{#1}}", 1],
}


def read_note(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm) or {}
    if not meta.get("category"):
        return None
    return {"title": str(meta.get("title", path.stem)), "subtitle": meta.get("subtitle", ""),
            "category": meta["category"], "body": body, "slug": path.stem, "folder": path.parent.name}


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_notes():
    order = load_yaml(REPO / "_data/notes_order.yml")
    by_part = {}
    for cat, folder, _ in PARTS:
        notes = [n for n in (read_note(p) for p in sorted((NOTES / folder).glob("*.md"))) if n and n["category"] == cat]
        rank = {s: i for i, s in enumerate(order.get(cat) or [])}
        notes.sort(key=lambda n: (rank.get(n["slug"], len(rank)), n["slug"]))
        by_part[cat] = notes
    return by_part


def svg_titles():
    out = {}
    for p in (NOTES / "assets").glob("*.svg"):
        m = re.search(r"<title[^>]*>(.*?)</title>", p.read_text(errors="ignore"), re.S)
        if m:
            out[p.name] = re.sub(r"\s+", " ", m.group(1)).strip()
    return out


def card_grid(notes, prefix, overlays):
    lines = ["::::{grid} 1 2 2 3", ":gutter: 3", ":class-container: chapter-cards", ""]
    for n in notes:
        ov = overlays.get(f"{n['folder']}/{n['slug']}", {})
        title = ov.get("title") or display_title(n["title"])
        hook = ov.get("hook") or re.sub(r"<[^>]+>", "", n["subtitle"] or "")
        nviz = len(ov.get("viz") or []) + sum(len((s or {}).get("viz") or []) for s in (ov.get("sections") or {}).values())
        meta = " · ".join(x for x in (f"{n['minutes']} min" if n.get("minutes") else "",
                                       f"{nviz} interactive" if nviz else "") if x)
        lines += [f":::{{grid-item-card}} {title}", f":link: {prefix}{n['folder']}/{n['slug']}", ":link-type: doc", "",
                  hook, ""] + ([f'<span class="card-meta">{meta}</span>', ""] if meta else []) + [":::", ""]
    return lines + ["::::", ""]


def topic_map(by_part):
    """tf-idf similarity between notes -> nodes + each note's strongest links, for the landing-page map."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    notes = [n for cat, _, _ in PARTS for n in by_part[cat]]
    texts = [re.sub(r"\(http[^)]*\)|\$[^$]*\$|<[^>]+>", " ", n["body"]) for n in notes]
    big = len(texts) >= 10   # small --only test builds can't use document-frequency cutoffs
    tfidf = TfidfVectorizer(stop_words="english", min_df=2 if big else 1, max_df=0.5 if big else 1.0,
                            sublinear_tf=True).fit_transform(texts)
    sim = (tfidf @ tfidf.T).toarray()
    def label(n):                    # short `map_label` from the overlay, else its title, else the note's
        ov = load_yaml(HERE / "overlays" / f"{n['folder']}/{n['slug']}.yml") or {}
        return ov.get("map_label") or ov.get("title") or display_title(n["title"])
    titles = {f"{n['folder']}/{n['slug']}": label(n) for n in notes}
    nodes = [{"id": f"{n['folder']}/{n['slug']}", "title": titles[f"{n['folder']}/{n['slug']}"], "part": n["category"],
              "url": f"notes/{n['folder']}/{n['slug']}.html"} for n in notes]
    edges = set()
    for i in range(len(notes)):
        sim[i, i] = 0
        for j in sim[i].argsort()[::-1][:min(3, len(notes) - 1)]:
            edges.add((min(i, int(j)), max(i, int(j))))
    links = [{"source": nodes[i]["id"], "target": nodes[j]["id"], "w": round(float(sim[i, j]), 3)} for i, j in sorted(edges)]
    parts = [{"id": cat, "title": title} for cat, _, title in PARTS]
    return {"nodes": nodes, "links": links, "parts": parts}, sim, [nd["id"] for nd in nodes]


PART_TITLE = {cat: title for cat, _, title in PARTS}
# italic words that are defined somewhere but too generic to link everywhere they are mentioned
GENERIC_TERMS = set("""process model models state states input output example problem system function value values
feature features training error goal agent agents action actions policy reward data sample samples method methods
approach task tasks interpretation consistent consistency mechanism primitives complete node nodes graph graphs
structure learning theory proof claim result results order level levels memory signal signals noise network
networks layer layers weight weights world object objects event events space time context concept concepts
explanation explanations representation representations domain domains query queries search cost risk""".split())


def chapter_title(n, overlays):
    return (overlays.get(f"{n['folder']}/{n['slug']}") or {}).get("title") or display_title(n["title"])


def chapter_hook(n, overlays):
    return (overlays.get(f"{n['folder']}/{n['slug']}") or {}).get("hook") or re.sub(r"<[^>]+>", "", n["subtitle"] or "")


def related_for(n, by_id, overlays, sim, ids, k=3):
    """Explicit overlay `related:` chapters first, then the most similar chapters by text (tf-idf)."""
    nid = f"{n['folder']}/{n['slug']}"
    picks = [r for r in (overlays.get(nid) or {}).get("related") or [] if r in by_id and r != nid]
    if nid in ids:
        i = ids.index(nid)
        for j in sim[i].argsort()[::-1]:
            if ids[j] != nid and ids[j] not in picks:
                picks.append(ids[j])
            if len(picks) >= k:
                break
    out = []
    for r in picks[:k]:
        m = by_id[r]
        out.append({"href": f"../{m['folder']}/{m['slug']}.html", "title": chapter_title(m, overlays),
                    "hook": chapter_hook(m, overlays), "part": PART_TITLE[m["category"]]})
    return out


def html_text(s, limit=None):
    """Note markdown -> plain html for reference pages: links to text, emphasis dropped, $math$ -> \\(math\\)."""
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"(?<!\\)\*+", "", s)
    if limit and len(s) > limit:
        cut = s[:limit]
        if cut.count("$") % 2:                          # don't cut inside math
            cut = cut[:cut.rfind("$")]
        s = cut.rsplit(" ", 1)[0].rstrip(",;:-") + " ..."
    parts = re.split(r"(\$[^$]+\$)", s)
    return "".join("\\(" + html.escape(x[1:-1], quote=False) + "\\)" if x.startswith("$") and x.endswith("$") and len(x) > 1
                   else html.escape(x, quote=False) for x in parts)


def glossary_page(defs):
    """All '- *term* - definition' lines in the notes, A-Z, filterable, each linking to where it is defined."""
    groups = {}
    for d in defs:
        groups.setdefault(d["term"].lower(), []).append(d)
    keys = sorted(groups, key=lambda k: re.sub(r"^[^a-z0-9]+", "", k))
    letters = {}
    for k in keys:
        c = (re.sub(r"^[^a-z0-9]+", "", k)[:1] or "#").upper()
        letters.setdefault(c if c.isalpha() else "#", []).append(k)
    nav = "".join(f'<a href="#g-{c.lower() if c != "#" else "sym"}">{c}</a>' for c in letters)
    body = []
    for c, ks in letters.items():
        body.append(f'<section class="glossary-letter" id="g-{c.lower() if c != "#" else "sym"}"><h3>{c}</h3>')
        for k in ks:
            ds = groups[k]
            where = " · ".join(f'<a href="{d["href"]}">{html.escape(d["chapter"])}</a>' for d in ds[:4])
            body.append(f'<div class="glossary-entry"><span class="g-term">{html.escape(ds[0]["term"])}</span> '
                        f'<span class="g-def">{html_text(ds[0]["text"], 190)}</span> <span class="g-where">{where}</span></div>')
        body.append("</section>")
    return "\n".join([
        "# Glossary", "",
        f"Every term the notes define with a *term* - definition line ({len(keys)} terms). "
        "Each entry links to the place where it is defined; filter to find one quickly.", "",
        "```{raw} html",
        f'<div class="glossary mathjax_process"><input class="glossary-filter" type="search" placeholder="Filter terms" '
        f'aria-label="Filter the glossary"><nav class="glossary-nav">{nav}</nav>' + "".join(body) + "</div>",
        "```", "", "{.mathjax-loader}", "$\\,$", ""])


def figures_page(vizzes, by_part, overlays):
    """Every interactive figure in the book, by part, linking to where it appears."""
    lines = ["# Interactive figures", "",
             "Every interactive figure in the book, grouped by part. Each one illustrates a single idea from its "
             "chapter; follow the link to see it in context.", ""]
    for cat, folder, title in PARTS:
        items = [v for v in vizzes if v["category"] == cat]
        if not items:
            continue
        lines += [f"## {title}", "", "```{raw} html", '<div class="figure-index mathjax_process">']
        for v in items:
            first = re.split(r"(?<=[.])\s", re.sub(r"\s+", " ", v["caption"].strip()))[0]
            name = v["name"].replace("-", " ")
            lines.append(f'<a class="fig-item" href="{v["href"]}"><span class="fig-name">{html.escape(name[:1].upper() + name[1:])}</span>'
                         f'<span class="fig-where">{html.escape(v["chapter"])} · {html.escape(v["section"] or "top")}</span>'
                         f'<span class="fig-desc">{html_text(first, 220)}</span></a>')
        lines += ["</div>", "```", ""]
    lines += ["{.mathjax-loader}", "$\\,$", ""]
    return "\n".join(lines)


def paths_html(paths, by_id, overlays):
    """Reading paths on the start page (from overlays/_landing.yml `paths:`)."""
    out = []
    for pth in paths or []:
        cs = [c for c in pth["chapters"] if c in by_id]
        names = [chapter_title(by_id[c], overlays) for c in cs]
        names = [f"{nm} ({PART_TITLE[by_id[c]['category']].lower()})" if names.count(nm) > 1 else nm for nm, c in zip(names, cs)]
        steps = "".join(f'<li><a href="notes/{by_id[c]["folder"]}/{by_id[c]["slug"]}.html">{html.escape(nm)}</a></li>'
                        for c, nm in zip(cs, names))
        out.append(f'<div class="path"><p class="path-title">{html.escape(pth["title"])}</p>'
                   f'<p class="path-desc">{html.escape(re.sub(chr(10), " ", pth.get("description", "")).strip())}</p>'
                   f'<ol class="path-steps">{steps}</ol></div>')
    return '<div class="paths">' + "".join(out) + "</div>"


def write_config(title):
    cfg = {
        "title": "", "author": "Chandan Singh", "copyright": "2026", "logo": "logo.png",
        "only_build_toc_files": True,
        "exclude_patterns": ["_build", "**.ipynb_checkpoints"],
        "execute": {"execute_notebooks": "off"},
        "parse": {"myst_enable_extensions": ["dollarmath", "amsmath", "colon_fence", "attrs_block", "attrs_inline",
                                              "deflist", "substitution", "html_image", "strikethrough"]},
        "repository": {"url": "https://github.com/csinva/csinva.github.io", "path_to_book": "_notes"},
        "html": {"use_repository_button": True, "home_page_in_navbar": False,
                 "extra_footer": "<p>Built from the markdown notes at <a href='https://csinva.io/'>csinva.io</a>. "
                                 "Many images are taken from resources on the web.</p>"},
        "sphinx": {"config": {
            "html_title": title,
            "numfig": False,                     # figures are numbered per page by transform.py
            "myst_heading_anchors": 3,
            "suppress_warnings": ["myst.header", "myst.xref_missing", "misc.highlighting_failure", "myst.strikethrough"],
            "mathjax3_config": {"tex": {"macros": MACROS}},
            "html_static_path": ["_static"],
            "html_css_files": [FONTS, "css/book.css"],
            "html_js_files": [CDN_D3, CDN_MERMAID, "js/nb-viz.js"] + sorted(f"js/viz/{p.name}" for p in (HERE / "static/js/viz").glob("*.js")),
            "html_theme_options": {
                "logo": {"image_light": "logo.png", "image_dark": "logo_dark.png"},
                "repository_url": "https://github.com/csinva/csinva.github.io",
                "use_repository_button": True, "use_download_button": False, "use_fullscreen_button": False,
                "show_toc_level": 2, "home_page_in_toc": True, "show_navbar_depth": 1,
                "navigation_with_keys": False,
            },
        }},
    }
    (SRC / "_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-src", action="store_true", help="keep the generated MyST sources in src/ for inspection")
    ap.add_argument("--only", nargs="*", help="build only these notes (folder/slug), for quick iteration")
    ap.add_argument("--no-build", action="store_true", help="only generate src/, don't run jupyter-book")
    ap.add_argument("--src-dir", help="where to write the generated MyST sources (default: src/)")
    ap.add_argument("--out-dir", help="where to put _build/ (default: this folder). Use a scratch dir for previews "
                                      "so parallel builds don't overwrite the real book")
    args = ap.parse_args()
    global SRC, OUT
    if args.src_dir:
        SRC = pathlib.Path(args.src_dir).resolve()
    out_root = pathlib.Path(args.out_dir).resolve() if args.out_dir else ROOT
    OUT = out_root / "_build"

    by_part = collect_notes()
    if args.only:
        by_part = {c: [n for n in ns if f"{n['folder']}/{n['slug']}" in args.only] for c, ns in by_part.items()}
    note_ids = {f"{n['folder']}/{n['slug']}" for ns in by_part.values() for n in ns}
    if not note_ids:
        sys.exit(f"no notes matched --only {args.only} (pass each note as its own argument, e.g. --only stat/stat ml/nlp)")
    overlays = {nid: load_yaml(HERE / "overlays" / f"{nid}.yml") for nid in note_ids}
    titles = svg_titles()

    shutil.rmtree(SRC, ignore_errors=True)
    (SRC / "notes").mkdir(parents=True)
    shutil.copytree(NOTES / "assets", SRC / "notes/assets")
    shutil.copytree(HERE / "static", SRC / "_static")
    for logo in ("logo.png", "logo_dark.png"):
        shutil.copy(ROOT / logo, SRC / logo)
        shutil.copy(ROOT / logo, SRC / "_static" / logo)

    tm, sim, sim_ids = topic_map(by_part)
    by_id = {f"{n['folder']}/{n['slug']}": n for ns in by_part.values() for n in ns}

    # first pass: collect every term definition, so mentions on other pages can link to it
    found = {}
    for cat, folder, _ in PARTS:
        for n in by_part[cat]:
            nid = f"{folder}/{n['slug']}"
            pre = Context(note_path=nid, note_urls=note_ids, svg_titles=titles, note_dir=None)
            note_to_myst(n, overlays[nid], pre)
            for d in pre.defs:
                found.setdefault(d["term"].lower(), []).append(dict(d, page=nid, chapter=chapter_title(n, overlays)))
    # link only distinctive terms: not too short, not generic, defined in at most two places
    term_links = {k: ds[0] for k, ds in found.items()
                  if len(k) >= 5 and len(ds) <= 2 and k not in GENERIC_TERMS and not re.search(r"\d", k)
                  and not ds[0]["page"].startswith("research_ovws/")}   # research definitions are paper-specific
    # distinctive multi-word terms are also linked where they appear in plain text
    plain = sorted((k for k in term_links if (" " in k or "-" in k) and len(k) >= 8 and not re.search(r"[()=:]", k)),
                   key=len, reverse=True)
    term_pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(k) for k in plain) + r")(?![\w-])") if plain else None
    drift, all_defs, all_viz = [], [], []
    for cat, folder, part_title in PARTS:
        for i, n in enumerate(by_part[cat], 1):
            nid = f"{folder}/{n['slug']}"
            words = len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\$[^$]*\$|\([^)]*https?://[^)]*\)|<[^>]+>", " ", n["body"])))
            n.update(part_title=part_title, index=i, count=len(by_part[cat]), minutes=max(1, round(words / 220)),
                     related=related_for(n, by_id, overlays, sim, sim_ids))
            ctx = Context(note_path=nid, note_urls=note_ids, svg_titles=titles, note_dir=NOTES / folder,
                          term_links=term_links, term_pattern=term_pattern)
            text = note_to_myst(n, overlays[nid], ctx)
            chap = chapter_title(n, overlays)
            all_defs += [dict(d, chapter=chap, href=f"../notes/{folder}/{n['slug']}.html#{d['anchor']}") for d in ctx.defs]
            all_viz += [dict(v, chapter=chap, category=cat, href=f"../notes/{folder}/{n['slug']}.html#{v['anchor']}") for v in ctx.vizzes]
            dest = SRC / "notes" / folder / f"{n['slug']}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            drift += [f"{nid}: {w}" for w in ctx.warnings]

    # part landing pages
    (SRC / "parts").mkdir()
    toc = {"format": "jb-book", "root": "intro", "parts": []}
    for cat, folder, title in PARTS:
        if not by_part[cat]:
            continue
        ov = load_yaml(HERE / "overlays/_parts" / f"{cat}.yml")
        lines = [f"# {ov.get('title', title)}", ""]
        lines += ["```{raw} html", '<div class="sidebars-open" hidden></div>', "```", ""]   # part overview pages keep both sidebars open
        if ov.get("intro"):
            for p in re.split(r"\n\s*\n", ov["intro"].strip()):
                lines += ["{.part-intro}", re.sub(r"\s+", " ", p.strip()), ""]
        if len(by_part[cat]) > 2:
            params = html.escape(json.dumps({"data": "_static/data/topic_map.json", "focus": cat}), quote=True)
            lines += ["```{raw} html", f'<figure class="nb-viz-figure part-map"><div class="nb-viz" data-viz="topic-map" data-params="{params}"></div>',
                      f'<figcaption>The chapters of this part, linked by shared vocabulary. Faded dots are the closest chapters in other parts.</figcaption></figure>',
                      "```", ""]
        lines += card_grid(by_part[cat], "../notes/", overlays)
        (SRC / "parts" / f"{cat}.md").write_text("\n".join(lines))
        toc["parts"].append({"caption": title, "chapters": [{"file": f"parts/{cat}", "title": "Overview"}] +
                             [{"file": f"notes/{folder}/{n['slug']}"} for n in by_part[cat]]})
    (SRC / "reference").mkdir()
    (SRC / "reference/glossary.md").write_text(glossary_page(all_defs))
    (SRC / "reference/figures.md").write_text(figures_page(all_viz, by_part, overlays))
    toc["parts"].append({"caption": "Reference", "chapters": [{"file": "reference/glossary"}, {"file": "reference/figures"}]})
    (SRC / "_toc.yml").write_text(yaml.safe_dump(toc, sort_keys=False))

    # start page
    land = load_yaml(HERE / "overlays/_landing.yml")
    (SRC / "_static/data").mkdir(parents=True, exist_ok=True)
    (SRC / "_static/data/topic_map.json").write_text(json.dumps(tm))
    lines = ["# " + land.get("title", "Notes on ML, AI, and neuroscience"), ""]
    lines += ["```{raw} html", '<div class="sidebars-open" hidden></div>', "```", ""]   # the start page keeps both sidebars open
    if land.get("lead"):
        lines += ["```{raw} html", f'<p class="landing-lead">{land["lead"].strip()}</p>', "```", ""]
    for p in re.split(r"\n\s*\n", (land.get("intro") or "").strip()):
        if p.strip():
            lines += [re.sub(r"\s+", " ", p.strip()), ""]
    if land.get("paths"):
        lines += ["## Where to start", "", re.sub(r"\s+", " ", land.get("paths_intro", "")).strip(), "",
                  "```{raw} html", paths_html(land["paths"], by_id, overlays), "```", ""]
    lines += ["## How the chapters connect", ""]
    if land.get("map_intro"):
        lines += [re.sub(r"\s+", " ", land["map_intro"].strip()), ""]
    lines += ["```{raw} html",
              '<figure class="nb-viz-figure landing-map"><div class="nb-viz" data-viz="topic-map" '
              'data-params="{&quot;data&quot;: &quot;_static/data/topic_map.json&quot;}"></div>',
              f'<figcaption>{land.get("map_caption", "Every chapter, linked to the chapters it shares the most vocabulary with.")}'
              '</figcaption></figure>', "```", ""]
    for cat, folder, title in PARTS:
        if by_part[cat]:
            ov = load_yaml(HERE / "overlays/_parts" / f"{cat}.yml")
            lines += [f"## [{title}](parts/{cat}.md)", ""]
            if ov.get("blurb"):
                lines += [re.sub(r"\s+", " ", ov["blurb"].strip()), ""]
            lines += card_grid(by_part[cat], "notes/", overlays)
    (SRC / "intro.md").write_text("\n".join(lines))
    write_config(land.get("html_title", "ML-AI Guide"))

    if drift:
        print(f"\n{len(drift)} overlay keys no longer match the notes (run book/check_overlays.py):")
        print("\n".join("  " + d for d in drift))
    if args.no_build:
        return

    shutil.rmtree(OUT, ignore_errors=True)
    res = subprocess.run(["jupyter-book", "build", str(SRC), "--path-output", str(out_root)], capture_output=True, text=True)
    log = res.stdout + res.stderr
    (out_root / "last_build.log" if args.out_dir else HERE / "last_build.log").write_text(log)
    warns = [l for l in log.split("\n") if "WARNING" in l or "ERROR" in l]
    print(f"jupyter-book exit {res.returncode}; {len(warns)} warnings/errors (book/last_build.log)")
    for w in warns[:25]:
        print("  " + w.replace(str(SRC) + "/", "")[:220])

    for leftover in ("html/_sources", ".doctrees", "html/.buildinfo", "jupyter_execute"):
        p = OUT / leftover
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    if not args.keep_src:
        shutil.rmtree(SRC, ignore_errors=True)


if __name__ == "__main__":
    main()

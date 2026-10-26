"""Turn one kramdown note from _notes/ into a MyST page for the jupyter-book.

The notes stay the source of truth. Everything the book adds on top of them (an intro,
short lead-ins for sections, figure captions, interactive widgets) comes from an overlay
file (see AGENTS.md) and is merged in here, keyed by section header text and image file
name, so editing a note never requires touching the book code.

Main entry point: `note_to_myst(note, overlay, ctx) -> str`.
"""
import html
import json
import re
import urllib.parse
from dataclasses import dataclass, field

LIST_ITEM = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+]|\d+\.)(?P<space>[ \t]+)(?P<rest>.*)$")
FENCE = re.compile(r"^\s*(?:(?:[-*+]|\d+\.)\s+)?(```|~~~)")   # fences can open after a list marker
HEADER = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
IMG_HTML = re.compile(r"<img\b[^>]*>", re.I)
IMG_MD = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\"]+?)(?:\s+\"[^\"]*\")?\)")   # src may contain spaces
ACRONYMS = {w.lower(): w for w in (
    "AI ML DL RL NLP CS OS STS GPU CPU LLM LLMs UCSF ICU CNN CNNs RNN RNNs LSTM GAN GANs VAE VAEs PCA ICA NMF LDA "
    "SVM SVMs MDP MDPs POMDP POMDPs RAG ATE SEM SCM DAG DAGs HMM HMMs CRF EM MLE MAP MCMC SGD OLS KL MDL BERT GPT "
    "PPO DQN UMAP IID ROC AUC FDR ANOVA RCT RCTs LSH ANN SQL CDI EHR fMRI EEG LIF").split()}
ACRONYMS["map"] = "map"            # 'map' is usually the english word in headers


def display_title(title):
    """'linear algebra' -> 'Linear algebra', 'llms' -> 'LLMs', 'rl, decisions' -> 'RL, decisions'."""
    words = re.split(r"(\W+)", title.strip())
    t = "".join(ACRONYMS.get(w.lower(), w) for w in words)
    return t[:1].upper() + t[1:]


def capitalize_first(text):
    """Book headers start with a capital letter ('basics' -> 'Basics'); the notes keep their own style."""
    m = re.search(r"[A-Za-z]", text)
    if not m or (m.start() > 0 and text[:m.start()].strip() not in ("", "[")):
        return text
    word = re.match(r"[A-Za-z]+", text[m.start():]).group(0)
    if word.lower() in ACRONYMS:
        return text[:m.start()] + ACRONYMS[word.lower()] + text[m.start() + len(word):]
    return text[:m.start()] + text[m.start()].upper() + text[m.start() + 1:]


def header_key(text):
    """Normalized header text used to match overlay `sections:` keys (ignores case, markup, links)."""
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    t = re.sub(r"[*_`]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def indent_of(line):
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip(" "))


def content_indent(line):
    """Column where a list item's text starts (nested blocks line up there); else the line's own indent."""
    m = LIST_ITEM.match(line.expandtabs(4))
    if not m:
        return indent_of(line)
    return len(m.group("indent")) + len(m.group("marker")) + len(m.group("space"))


def split_display_math(expr):
    """True for a real multi-line equation: a top-level \\\\ or an align-type environment."""
    flat = re.sub(r"\\begin\{([a-zA-Z]*matrix|cases|array|aligned|split)\*?\}.*?\\end\{\1\*?\}", "", expr, flags=re.S)
    return bool(re.search(r"\\\\|\\begin\{(align|eqnarray|gather|multline)", flat))


@dataclass
class Figure:
    src: str            # path as written in the note (relative to the note)
    alt: str
    width: str = ""     # e.g. "400px" or "60%"


def parse_img_tag(tag):
    def get(a):
        m = re.search(a + r'\s*=\s*"([^"]*)"', tag, re.I) or re.search(a + r"\s*=\s*'([^']*)'", tag, re.I)
        return m.group(1) if m else ""
    width = ""
    m = re.search(r"(?<![-\w])width:\s*([\d.]+(?:px|%))", get("style"))
    if m:
        width = m.group(1)
    if not width and get("class"):
        width = {"medium_image": "60%", "big_image": "80%", "full_image": "100%"}.get(get("class").split()[0], "")
    return Figure(get("src"), html.unescape(get("alt")), width)


def junk_alt(alt):
    """Alt texts like 'Screen Shot 2018-07-31 at 10.07.40 PM' or a bare file stem carry no meaning."""
    return (not alt.strip()) or bool(re.search(r"screen ?shot|\d{4}-\d\d-\d\d", alt, re.I)) \
        or bool(re.fullmatch(r"[\w\-\.]+", alt.strip()))


@dataclass
class Context:
    note_path: str                    # e.g. "ml/classification"
    note_urls: set                    # {"ml/classification", ...}: notes that are in the book
    svg_titles: dict                  # image file name -> <title> of the svg (fallback caption)
    note_dir: object = None           # pathlib.Path of the note's folder, to skip images that don't exist
    warnings: list = field(default_factory=list)
    used_sections: set = field(default_factory=set)
    used_figures: set = field(default_factory=set)
    n_viz: int = 0
    n_fig: int = 0
    section: str = ""                 # header text of the section being converted
    defs: list = field(default_factory=list)    # term definitions found (for the glossary)
    term_links: dict = field(default_factory=dict)  # term (lower) -> definition elsewhere, for linking mentions
    term_pattern: object = None       # compiled regex of distinctive multi-word terms (plain-text mentions)
    linked: set = field(default_factory=set)    # terms already linked on this page (first mention only)
    outline: list = field(default_factory=list) # top-level sections: title, anchor, lead, figure/widget counts
    vizzes: list = field(default_factory=list)  # widgets placed (for the interactive-figures index)


def convert_links(line, ctx):
    """Links to other notes on csinva.io become relative links between book pages."""
    def repl(m):
        text, url = m.group(1), m.group(2)
        mm = re.match(r"https?://csinva\.io/notes/([a-z_]+)/([a-z_0-9]+)\.html(#.*)?$", url) or \
            re.match(r"(?:\.\./)?([a-z_]+)/([a-z_0-9]+)\.html(#.*)?$", url)
        if mm and f"{mm.group(1)}/{mm.group(2)}" in ctx.note_urls:
            here = ctx.note_path.split("/")[0]
            target = f"{mm.group(2)}.md" if mm.group(1) == here else f"../{mm.group(1)}/{mm.group(2)}.md"
            return f"[{text}]({target})"
        return m.group(0)
    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", repl, line)


NAMED_COLORS = {"teal": "green", "green": "green", "forestgreen": "green", "blue": "blue", "navyblue": "blue",
                "cornflowerblue": "blue", "cyan": "blue", "red": "pink", "magenta": "pink", "purple": "purple",
                "violet": "purple", "orange": "orange", "brown": "orange", "olive": "yellow"}


def theme_colors(line):
    """\\textcolor{teal}{..} / \\color{red} written in the notes -> the theme's color tokens (work in dark mode)."""
    def repl(m):
        tok = NAMED_COLORS.get(m.group(2).lower())
        return f"\\{m.group(1)}{{var(--nb-c-{tok})}}" if tok else m.group(0)
    return re.sub(r"\\(textcolor|color)\{([A-Za-z]+)\}", repl, line)


def convert_math(line):
    """kramdown math conventions -> MyST dollarmath.

    `$$x$$` in a sentence or table cell is inline math in kramdown but display math in MyST, so it
    becomes `$x$`. A list item that is only `$$x$$` shows inline on the site, so single-line ones
    become `$x$` too; real multi-line equations stay display math.
    """
    m = LIST_ITEM.match(line)
    body = m.group("rest") if m else line.strip()
    whole = re.fullmatch(r"\$\$(.+?)\$\$\s*", body)
    if whole:
        expr = whole.group(1).strip()
        if m and not split_display_math(expr):
            return f"{m.group('indent')}{m.group('marker')}{m.group('space')}${expr}$"
        if not m and indent_of(line) > 0 and not split_display_math(expr):
            # a lone one-line $$..$$ continuing a list item's text: MyST only makes display math from a
            # block of its own, so inside a paragraph it would show raw dollars; render it inline instead
            return line[:indent_of(line)] + f"${expr}$"
        return line
    return re.sub(r"\$\$(.+?)\$\$", lambda mm: "$" + mm.group(1).strip() + "$", line)


def figure_block(fig, indent, ctx, overlay_figs):
    """Numbered MyST figure for one image, indented to sit inside a list item when needed."""
    name = urllib.parse.unquote(fig.src.split("/")[-1])
    if ctx.note_dir is not None and not fig.src.startswith("http") and \
            not (ctx.note_dir / urllib.parse.unquote(fig.src)).exists():
        ctx.warnings.append(f"image file missing, left out of the book: {fig.src}")
        return []
    if name in overlay_figs:
        caption = overlay_figs[name] or ""
        ctx.used_figures.add(name)
    else:
        caption = ctx.svg_titles.get(name) or ("" if junk_alt(fig.alt) else fig.alt)
    pad = " " * indent
    alt = re.sub(r"\s+", " ", caption or fig.alt or name)
    if caption:                                     # figures are numbered per page (sphinx numfig is global)
        ctx.n_fig += 1
        caption = f"[Figure {ctx.n_fig}.]{{.fig-num}} " + caption.strip()
    out = [f"{pad}```{{figure}} {urllib.parse.unquote(fig.src)}"]
    if fig.width:
        out.append(f"{pad}:width: {fig.width}")
    out += [f"{pad}:alt: {alt[:200]}", f"{pad}:class: note-figure", ""]
    if caption:
        out.append(pad + re.sub(r"\s+", " ", caption.strip()))
    out.append(f"{pad}```")
    return out


def convert_images(line, ctx, overlay_figs):
    """Lines with images (html or markdown) -> the line's text plus figure directive(s) under it."""
    if line.lstrip().startswith("|"):          # images in table cells stay plain images
        return [IMG_HTML.sub(lambda m: "![{0.alt}]({0.src})".format(parse_img_tag(m.group(0))), line)]
    figs = []
    def grab_html(m):
        figs.append(parse_img_tag(m.group(0)))
        return " "
    def grab_md(m):
        figs.append(Figure(m.group("src"), m.group("alt")))
        return " "
    orig = line.expandtabs(4)
    text = IMG_MD.sub(grab_md, IMG_HTML.sub(grab_html, orig))
    figs = [f for f in figs if f.src]
    if not figs:
        return [line]
    m = LIST_ITEM.match(orig)          # bullet/indent from the original line (images became spaces in `text`)
    rest = re.sub(r"[\s\-:,]+$", "", text[len(m.group(0)) - len(m.group("rest")):] if m else text).strip()
    if len(figs) >= 2 and not rest and not m:          # an image group with no text: figures side by side
        ind = indent_of(orig)
        pad = " " * ind
        out = [f"{pad}::::{{grid}} {min(len(figs), 3)}", f"{pad}:gutter: 2", ""]
        for f in figs:
            f.width = ""
            out += [f"{pad}:::{{grid-item}}"] + figure_block(f, ind, ctx, overlay_figs) + [f"{pad}:::", ""]
        return out + [f"{pad}::::", ""]
    out = []
    if m:
        first = f"{m.group('indent')}{m.group('marker')} "
        ind = len(first)
        blocks = []
        for f in figs:
            blocks += figure_block(f, ind, ctx, overlay_figs)
        if rest:
            out = [first + rest] + blocks
        elif blocks:
            blocks[0] = first + blocks[0].lstrip()
            out = blocks
    else:
        ind = indent_of(orig)
        if rest:
            out += [" " * ind + rest, ""]
        for f in figs:
            out += figure_block(f, ind, ctx, overlay_figs) + [""]
    return out


def convert_details(text):
    """<details><summary>T</summary>...</details> -> MyST dropdown (MyST doesn't parse markdown inside html)."""
    pat = re.compile(r"(?P<lead>^[ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?)<details[^>]*>\s*<summary[^>]*>(?P<title>.*?)</summary>"
                     r"(?P<body>.*?)</details>", re.S | re.M)
    def repl(m):
        lead = m.group("lead").expandtabs(4)
        pad = " " * len(lead)
        body = re.sub(r"</?p>", "", m.group("body"))
        lines = [l.strip() for l in body.strip().split("\n") if l.strip()]
        title = re.sub("<[^>]+>", "", m.group("title")).strip()
        return "\n".join([f"{lead}::::{{dropdown}} {title}"] + [pad + l for l in lines] + [pad + "::::"])
    return pat.sub(repl, text)


DEF = re.compile(r"^(?P<lead>\s*(?:[-*+]|\d+\.)\s+)(?P<term>\*\*(?P<b>[^*]{2,40})\*\*|\*(?P<i>[^*]{2,40})\*)"
                 r"(?P<sep>\s*(?:\$[^$]{1,30}\$\s*)?(?:\([^)]{0,40}\)\s*)?(?:-|:|=)\s*)(?P<text>\S.{6,})$")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "x"


def mark_definition(line, ctx):
    """'- *term* - definition' lines get an anchor and are collected for the glossary page."""
    m = DEF.match(line)
    if not m:
        return line
    term = (m.group("b") or m.group("i")).strip()
    if re.search(r"[$<\[]", term) or term.lower() in ("ex", "e.g.", "note", "eg", "i.e."):
        return line
    anchor = "def-" + slugify(term)
    used = {d["anchor"] for d in ctx.defs}
    k = 2
    while anchor in used:
        anchor = f"def-{slugify(term)}-{k}"; k += 1
    ctx.defs.append({"term": term, "text": m.group("text").strip(), "anchor": anchor, "section": ctx.section})
    return f'{m.group("lead")}<span id="{anchor}" class="def-anchor"></span>{line[len(m.group("lead")):]}'


ITALIC = re.compile(r"(?<![*\w\\])\*(?!\s)([^*\n$]{4,40}?)(?<!\s)\*(?![*\w])")


def plain_def(text):
    """A definition as plain text for a hover popover (no markdown, math dropped)."""
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    s = re.sub(r"<[^>]+>|\*+", "", s)
    s = re.sub(r"\$[^$]*\$", "...", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:220].rsplit(" ", 1)[0] + " ..." if len(s) > 220 else s


def link_terms(line, ctx):
    """First italic mention of a term defined elsewhere -> link to its definition with a hover popover."""
    if not ctx.term_links or "def-anchor" in line or line.lstrip().startswith("|"):
        return line
    def repl(m):
        term = m.group(1)
        d = ctx.term_links.get(term.lower())
        pre = line[:m.start()]
        if not d or term.lower() in ctx.linked or pre.count("[") > pre.count("]") or pre.count("$") % 2:
            return m.group(0)
        if d["page"] == ctx.note_path and d["section"] == ctx.section:
            return m.group(0)                       # defined right here
        ctx.linked.add(term.lower())
        folder, slug = d["page"].split("/")
        here = ctx.note_path.split("/")[0]
        href = (f"#{d['anchor']}" if d["page"] == ctx.note_path else
                f"{slug}.html#{d['anchor']}" if folder == here else f"../{folder}/{slug}.html#{d['anchor']}")
        tip = html.escape(f"{plain_def(d['text'])} · {d['chapter']}", quote=True)
        return f'<a class="term-link" href="{href}" data-def="{tip}"><em>{html.escape(term, quote=False)}</em></a>'
    line = ITALIC.sub(repl, line)
    return link_terms_plain(line, ctx) if ctx.term_pattern is not None else line


def link_terms_plain(line, ctx):
    """First plain-text mention of a distinctive multi-word term defined on another page -> link (never inside
    math, code, links or html; the rest of the line is processed again after each inserted link)."""
    masked = re.sub(r"\$[^$]*\$|`[^`]*`|\[[^\]]*\]\([^)]*\)|<[^>]+>.*?</a>|<[^>]+>|https?://\S+",
                    lambda m: "\0" * len(m.group(0)), line)
    for m in ctx.term_pattern.finditer(masked.lower()):
        term = m.group(0)
        d = ctx.term_links.get(term)
        if not d or term in ctx.linked or d["page"] == ctx.note_path:
            continue
        ctx.linked.add(term)
        folder, slug = d["page"].split("/")
        href = f"{slug}.html#{d['anchor']}" if folder == ctx.note_path.split("/")[0] else f"../{folder}/{slug}.html#{d['anchor']}"
        tip = html.escape(f"{plain_def(d['text'])} · {d['chapter']}", quote=True)
        return line[:m.start()] + f'<a class="term-link" href="{href}" data-def="{tip}">{line[m.start():m.end()]}</a>' + \
            link_terms_plain(line[m.end():], ctx)
    return line


def viz_block(spec, ctx):
    """Raw-html container for an interactive widget (static/js/nb-viz.js) with a numbered caption."""
    ctx.n_viz += 1
    anchor = f"viz-{spec['name']}" + ("" if not any(v["anchor"] == f"viz-{spec['name']}" for v in ctx.vizzes) else f"-{ctx.n_viz}")
    ctx.vizzes.append({"name": spec["name"], "caption": spec.get("caption", ""), "anchor": anchor, "section": ctx.section})
    params = html.escape(json.dumps(spec.get("params", {})), quote=True)
    cap = html.escape(re.sub(r"\s+", " ", spec.get("caption", "")).strip(), quote=False)
    cap = re.sub(r'&lt;span class="(c(?:blue|orange|green|pink|purple|yellow))"&gt;(.*?)&lt;/span&gt;',
                 r'<span class="\1">\2</span>', cap)          # colored words matching an equation
    cap = re.sub(r"\$([^$]+)\$", r"\\(\1\\)", cap)        # inline math in captions
    return ["", "```{raw} html",
            f'<figure class="nb-viz-figure" id="{anchor}"><div class="nb-viz" data-viz="{spec["name"]}" data-params="{params}"></div>',
            f'<figcaption class="mathjax_process"><span class="nb-viz-label">Interactive {ctx.n_viz}.</span> {cap}</figcaption></figure>',
            "```", ""]


def lead_block(text):
    out = [""]
    for p in re.split(r"\n\s*\n", text.strip()):
        if p.strip():
            out += ["{.section-lead}", re.sub(r"\s+", " ", p.strip()), ""]
    return out


def note_to_myst(note, overlay, ctx):
    """note: {title, subtitle, body} parsed from the kramdown file; overlay: dict from the overlay yaml."""
    overlay = overlay or {}
    sections = {header_key(k): v or {} for k, v in (overlay.get("sections") or {}).items()}
    overlay_figs = dict(overlay.get("figures") or {})

    body = note["body"]
    body = re.sub(r"^\s*\{:toc\}\s*$", "", body, flags=re.M)
    body = re.sub(r"^\s*\{:\s*[^}]*\}\s*$", "", body, flags=re.M)     # kramdown IALs on their own line
    body = convert_details(body)
    # <p float="left"> ... </p> groups of images were laid out side by side on the site: keep them together on
    # one line so convert_images turns them into a grid of figures
    def group(m):
        imgs = IMG_HTML.findall(m.group(2)) + [mm.group(0) for mm in IMG_MD.finditer(m.group(2))]
        return m.group(1) + " ".join(imgs) if len(imgs) >= 2 else m.group(0)
    body = re.sub(r"^([ \t]*)<p\s+float=[^>]*>(.*?)</p>", group, body, flags=re.S | re.M)
    # standalone <p ...> / </p> wrapper lines (used on the site to lay images side by side) would start a
    # raw-html block in MyST and swallow the figures inside them
    body = re.sub(r"^[ \t]*</?p(?:\s[^>]*)?>[ \t]*$", "", body, flags=re.M)
    # same for <span> layout wrappers and "<b>label</b><br>" lines above side-by-side images
    # (each figure gets its own caption instead)
    body = re.sub(r"^[ \t]*(?:</?span(?:\s[^>]*)?>|<b>[^<]*</b>\s*<br\s*/?>)[ \t]*$", "", body, flags=re.M)

    # widgets placed `after:` a bullet (matched by its text) go right after that bullet and its sub-items
    after = [dict(v) for s in sections.values() for v in (s.get("viz") or []) if v.get("after")]
    pending = []                                   # [indent of the matched bullet, spec]

    def flush(cur_indent):
        nonlocal pending
        keep = []
        for ind, spec in pending:
            if cur_indent is None or cur_indent <= ind:
                out.extend(viz_block(spec, ctx))
            else:
                keep.append((ind, spec))
        pending = keep

    out, fence, mermaid = [], False, None
    for line in body.split("\n"):
        if pending and not fence and mermaid is None and line.strip() and not FENCE.match(line):
            flush(indent_of(line))
        if after and not fence and mermaid is None and LIST_ITEM.match(line):
            for spec in list(after):
                if spec["after"].lower() in line.lower():
                    after.remove(spec)
                    ctx.used_after = getattr(ctx, "used_after", set()) | {spec["name"]}
                    pending.append((indent_of(line), spec))
        if mermaid is not None:                        # inside a ```mermaid block: render it as a diagram
            if FENCE.match(line):
                lead = re.match(r"^(\s*(?:(?:[-*+]|\d+\.)\s+)?)", mermaid[0].expandtabs(4)).group(1)
                pad = " " * len(lead)
                out += [lead + "```{raw} html", pad + '<div class="mermaid-wrap"><pre class="mermaid">']
                out += [pad + html.escape(l.strip()) for l in mermaid[1:] if l.strip()]
                out += [pad + "</pre></div>", pad + "```"]
                mermaid = None
            else:
                mermaid.append(line)
            continue
        if not fence and re.match(r"^\s*(?:(?:[-*+]|\d+\.)\s+)?```\s*mermaid\s*$", line):
            mermaid = [line]
            continue
        if FENCE.match(line):
            fence = not fence
            out.append(line)
            continue
        if fence:
            out.append(line)
            continue
        h = HEADER.match(line)
        if h:
            ctx.section = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h.group(2)).strip()
            if len(h.group(1)) == 1:                       # top-level section: stable anchor + outline entry
                anchor = f"sec-{ctx.note_path.split('/')[-1].replace('_', '-')}-{slugify(ctx.section)}"   # labels are book-wide
                while any(o["anchor"] == anchor for o in ctx.outline):
                    anchor += "-x"
                ctx.outline.append({"title": capitalize_first(ctx.section), "anchor": anchor, "lead": "",
                                    "fig0": ctx.n_fig, "viz0": ctx.n_viz})
                out += ["", f"({anchor})="]
            out.append("#" * min(len(h.group(1)) + 1, 6) + " " + capitalize_first(convert_links(h.group(2), ctx)))
            key = header_key(h.group(2))
            spec = sections.get(key)
            if spec and key not in ctx.used_sections:      # first header with this text only
                ctx.used_sections.add(key)
                if spec.get("lead"):
                    out += lead_block(spec["lead"])
                    if len(h.group(1)) == 1 and ctx.outline:
                        ctx.outline[-1]["lead"] = re.split(r"(?<=[.])\s", re.sub(r"\s+", " ", spec["lead"].strip()))[0]
                for v in spec.get("viz") or []:
                    if not v.get("after"):
                        out += viz_block(v, ctx)
            continue
        line = link_terms(mark_definition(convert_links(theme_colors(convert_math(line)), ctx), ctx), ctx)
        if IMG_HTML.search(line) or IMG_MD.search(line):
            out += convert_images(line, ctx, overlay_figs)
        else:
            out.append(line)

    flush(None)                                    # a matched bullet at the very end of the note
    ctx.warnings += [f"widget {v['name']!r}: no bullet contains {v['after']!r}" for v in after]

    title = overlay.get("title") or display_title(note["title"])
    head = []
    if note.get("part_title"):
        head += ["```{raw} html", f'<p class="breadcrumb"><a href="../../parts/{note["category"]}.html">{note["part_title"]}</a>'
                 f' &nbsp;·&nbsp; chapter {note["index"]} of {note["count"]}'
                 + (f' &nbsp;·&nbsp; {note["minutes"]} min read' if note.get("minutes") else "") + '</p>', "```", ""]
    head += [f"# {title}", ""]
    subtitle = overlay.get("subtitle") or note.get("subtitle")
    if subtitle:
        head += ["```{raw} html", f'<p class="note-subtitle">{subtitle}</p>', "```", ""]
    if overlay.get("intro"):
        head += ["```{admonition} About this chapter", ":class: chapter-intro", ""]
        for p in re.split(r"\n\s*\n", overlay["intro"].strip()):
            head += [re.sub(r"\s+", " ", p.strip()), ""]
        head += ["```", ""]
    for v in overlay.get("viz") or []:
        head += viz_block(v, ctx)
    if len(ctx.outline) >= 6:                        # long chapters: a collapsible outline of the sections
        rows = []
        for i, o in enumerate(ctx.outline):
            nxt = ctx.outline[i + 1] if i + 1 < len(ctx.outline) else {"fig0": ctx.n_fig, "viz0": ctx.n_viz}
            nf, nv = nxt["fig0"] - o["fig0"], nxt["viz0"] - o["viz0"]
            badges = (f'<span class="ol-badge">{nv} interactive</span>' if nv else "") + \
                     (f'<span class="ol-badge">{nf} figure{"s" if nf > 1 else ""}</span>' if nf else "")
            lead_txt = html.escape(re.sub(r"[$*]", "", re.sub(r"<[^>]+>", "", o["lead"])))
            lead = f'<span class="ol-lead">{lead_txt}</span>' if o["lead"] else ""
            name = html.escape(re.sub(r"[*`]", "", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", o["title"])))
            rows.append(f'<li><a href="#{o["anchor"]}">{name}</a>{badges}{lead}</li>')
        head += ["```{raw} html", f'<details class="chapter-outline"><summary>Chapter outline: {len(ctx.outline)} sections</summary>'
                 f'<ol>{"".join(rows)}</ol></details>', "```", ""]

    # sphinx only loads MathJax on pages with math nodes: make sure widget captions with $..$ still get it
    if "nb-viz-label" in "\n".join(out + head) and "$" in json.dumps(overlay) and "$" not in body:
        out += ["", "{.mathjax-loader}", "$\\,$", ""]

    if overlay.get("takeaways"):                   # "Key ideas" box closing the chapter
        out += ["", "```{admonition} Key ideas", ":class: chapter-takeaways", ""]
        out += ["- " + re.sub(r"\s+", " ", str(k).strip()) for k in overlay["takeaways"]]
        out += ["```", ""]

    if note.get("related"):                       # filled in by build_book.py (overlay `related` + text similarity)
        cards = "".join(f'<a class="related-card" href="{r["href"]}"><span class="related-part">{r["part"]}</span>'
                        f'<span class="related-title">{html.escape(r["title"])}</span>'
                        f'<span class="related-hook">{html.escape(r["hook"])}</span></a>' for r in note["related"])
        out += ["", "```{raw} html", f'<nav class="related-chapters"><p class="related-heading">Related chapters</p>'
                f'<div class="related-grid">{cards}</div></nav>', "```", ""]

    ctx.warnings += [f"overlay section not in note: {k!r}" for k in sections if k not in ctx.used_sections]
    ctx.warnings += [f"overlay figure not in note: {k!r}" for k in overlay_figs if k not in ctx.used_figures]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(head + out))
    return text.rstrip() + "\n"

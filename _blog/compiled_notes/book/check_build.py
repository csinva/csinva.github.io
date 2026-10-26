"""Sanity-check a built book: leaked markup in the visible text, and pages that lost content.

    cd _blog/compiled_notes && uv run python book/check_build.py [--build-dir _build/html]

For every chapter page it reports
- markup that should have been converted but shows up as text (```, {figure}, :width:, ::::, {raw} ...)
- the ratio of visible words on the page to words in the source note (a page far below 1.0 swallowed
  content, e.g. an unclosed fence or a raw-html block that broke out of its list)
Exit code 1 if anything looks wrong.
"""
import argparse
import html
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
LEAKS = [r"```", r"\{figure\}", r"\{raw\}", r"\{dropdown\}", r"\{admonition\}", r":width:", r":alt:", r"::::",
         r"\{\.section-lead\}", r"\{:toc\}", r"<img\b", r"</?details", r"\{grid"]


def visible_text(page):
    t = page.read_text(errors="ignore")
    m = re.search(r'<article class="bd-article"[^>]*>(.*?)</article>', t, re.S)
    t = m.group(1) if m else t
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", t, flags=re.S)
    t = re.sub(r'<div class="highlight.*?</div>', " ", t, flags=re.S)       # code blocks may contain anything
    t = re.sub(r"<pre\b.*?</pre>", " ", t, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", t))


def words(s):
    s = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^$\n]*\$", " ", s, flags=re.S)  # ignore math
    return len(re.findall(r"[A-Za-z]{3,}", s))


JSC = pathlib.Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc")


def check_widgets():
    """Syntax-check every widget with JavaScriptCore (macOS) so a broken file can't silently fail to register."""
    import subprocess
    if not JSC.exists():
        print("(no jsc: widget syntax not checked)")
        return 0
    bad = 0
    for js in sorted((HERE / "static/js/viz").glob("*.js")):
        r = subprocess.run([str(JSC), "-e", "var NB={register(){}};", str(js)], capture_output=True, text=True)
        err = (r.stdout + r.stderr).strip()
        if "SyntaxError" in err:
            bad += 1
            print(f"widget {js.name}: {err.splitlines()[0][:200]}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=str(HERE.parent / "_build/html"))
    args = ap.parse_args()
    build = pathlib.Path(args.build_dir)
    bad = 0
    for page in sorted((build / "notes").rglob("*.html")):
        rel = page.relative_to(build / "notes").with_suffix("")
        src = REPO / "_notes" / f"{rel}.md"
        text = visible_text(page)
        leaks = [p for p in LEAKS if re.search(p, text)]
        ratio = None
        if src.exists():
            body = src.read_text(errors="ignore").split("---", 2)[-1]
            body = re.sub(r"```.*?```", " ", body, flags=re.S)
            body = re.sub(r"\]\([^)]*\)|<[^>]+>|https?://\S+", " ", body)
            n_src = words(body)
            ratio = words(text) / max(n_src, 1)
        flag = leaks or (ratio is not None and ratio < 0.9)
        if flag:
            bad += 1
            print(f"{rel}: words page/source = {ratio:.2f}" if ratio is not None else f"{rel}:", "leaked:", leaks)
    bad += check_widgets()
    print(f"checked {len(list((build / 'notes').rglob('*.html')))} pages and the widgets, {bad} with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

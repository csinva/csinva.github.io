"""Check the overlays (book/overlays/**.yml) against the current notes.

    cd _blog/compiled_notes && uv run python book/check_overlays.py [--coverage] [note ...]

Reports, per note:
- DRIFT: overlay `sections:` keys whose header no longer exists in the note, `figures:` keys for images
  the note no longer uses, widgets (`viz: name`) that aren't registered in static/js/viz/
- VOICE: overlay text with phrases that read as generic or machine-written (see BANNED below)
- with --coverage: notes without an overlay or intro, and figures without a caption
Exit code 1 if there is drift or a voice problem, so it can gate a build.
"""
import argparse
import pathlib
import re
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_book import NOTES, PARTS, read_note  # noqa: E402
from transform import HEADER, IMG_HTML, IMG_MD, header_key, parse_img_tag  # noqa: E402

# words and constructions that make text sound generic; the overlays should read like the notes' author wrote them
BANNED = [r"\bdelve", r"\bcrucial", r"\bpivotal", r"\blandscape\b", r"\btapestry", r"\brealm\b", r"\bseamless",
          r"\bleverag(?!e score)", r"\bharness", r"\bunlock", r"\bdive into\b", r"\bjourney\b", r"\bnuanced\b", r"\bintricate",
          r"\bmultifaceted", r"\bparamount\b", r"\bunderscore", r"\bshowcas", r"\bgame.?changer", r"\bcutting.edge",
          r"\bin today's\b", r"\bit'?s worth noting\b", r"\bit is important to note\b", r"\bin this chapter,? we\b",
          r"\bwe will (?:explore|discuss|see)\b", r"\blet'?s\b", r"\bfascinating\b", r"\bpowerful\b", r"\bkey takeaway",
          r"\bnot only\b.*\bbut also\b", r"\bin conclusion\b", r"\bin summary\b", r"\boverall,", "—", r"!\s"]


def note_index(path):
    """Header keys and image file names actually used in a note."""
    text = path.read_text(errors="ignore")
    headers, images, fence = set(), set(), False
    for line in text.split("\n"):
        if re.match(r"^\s*(?:(?:[-*+]|\d+\.)\s+)?```", line):     # fences can open after a list marker
            fence = not fence
        if fence:
            continue
        h = HEADER.match(line)
        if h:
            headers.add(header_key(h.group(2)))
        for m in IMG_HTML.finditer(line):
            images.add(parse_img_tag(m.group(0)).src.split("/")[-1])
        for m in IMG_MD.finditer(line):
            images.add(m.group("src").split("/")[-1])
    return headers, {re.sub(r"%20", " ", i) for i in images}


def overlay_texts(ov):
    """All prose in an overlay (for the voice check)."""
    out = [ov.get(k, "") or "" for k in ("intro", "hook", "subtitle", "title")]
    for s in (ov.get("sections") or {}).values():
        s = s or {}
        out.append(s.get("lead", "") or "")
        out += [v.get("caption", "") for v in s.get("viz") or []]
    out += [v.get("caption", "") for v in ov.get("viz") or []]
    out += [str(c) for c in (ov.get("figures") or {}).values()]
    out += [str(k) for k in ov.get("takeaways") or []]
    return [t for t in out if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notes", nargs="*", help="limit to these notes (folder/slug)")
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()
    widgets = set()
    for js in (HERE / "static/js/viz").glob("*.js"):
        widgets |= set(re.findall(r'NB\.register\("([^"]+)"', js.read_text()))
    problems = 0
    for cat, folder, _ in PARTS:
        for path in sorted((NOTES / folder).glob("*.md")):
            n = read_note(path)
            nid = f"{folder}/{path.stem}"
            if not n or n["category"] != cat or (args.notes and nid not in args.notes):
                continue
            ovp = HERE / "overlays" / f"{nid}.yml"
            ov = yaml.safe_load(ovp.read_text()) if ovp.exists() else None
            headers, images = note_index(path)
            msgs = []
            if ov is None:
                if args.coverage:
                    msgs.append("coverage: no overlay file")
            else:
                for k in (ov.get("sections") or {}):
                    if header_key(k) not in headers:
                        msgs.append(f"DRIFT: section {k!r} not found in note")
                for k in (ov.get("figures") or {}):
                    if k not in images:
                        msgs.append(f"DRIFT: figure {k!r} not used in note")
                specs = list(ov.get("viz") or []) + [v for s in (ov.get("sections") or {}).values() for v in (s or {}).get("viz") or []]
                for v in specs:
                    if v.get("name") not in widgets:
                        msgs.append(f"DRIFT: widget {v.get('name')!r} is not registered in static/js/viz/")
                for t in overlay_texts(ov):
                    for b in BANNED:
                        m = re.search(b, t, re.I)
                        if m:
                            msgs.append(f"VOICE: {m.group(0)!r} in: {t.strip()[:90]!r}")
                if args.coverage:
                    if not ov.get("intro"):
                        msgs.append("coverage: no intro")
                    missing = sorted(i for i in images if i not in (ov.get("figures") or {}) and not i.endswith(".svg"))
                    if missing:
                        msgs.append(f"coverage: {len(missing)} raster figures without an overlay caption")
            hard = [m for m in msgs if m.startswith(("DRIFT", "VOICE"))]
            problems += len(hard)
            if msgs:
                print(nid)
                print("\n".join("  " + m for m in msgs))
    print(f"{problems} drift/voice problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

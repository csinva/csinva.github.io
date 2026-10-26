"""Add (or replace) widget placements in overlay files without reformatting them.

    uv run python book/tools/place_viz.py placements.yml

placements.yml is a list of
    - note: ml/deep_learning          # folder/slug
      section: basics                  # header text as in the note, or "_top" for the top of the page
      viz: {name: backprop-graph, params: {...}, caption: "..."}
A placement whose widget name already appears in the overlay replaces that entry. The overlay keeps its
comments and layout; the result is parsed again to make sure the placement landed.
"""
import pathlib
import re
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from transform import header_key  # noqa: E402


def item_lines(viz, indent):
    text = yaml.safe_dump([viz], sort_keys=False, allow_unicode=True, width=110).rstrip("\n")
    return [" " * indent + l for l in text.split("\n")]


def remove_viz(lines, name):
    """Drop an existing '- name: <name>' list item (and its continuation lines)."""
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)- name:\s*['\"]?" + re.escape(name) + r"['\"]?\s*$", l)
        if m:
            ind = len(m.group(1))
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > ind):
                j += 1
            del lines[i:j]
            return True
    return False


def block_end(lines, start, indent):
    j = start + 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent):
        j += 1
    while j > start + 1 and not lines[j - 1].strip():
        j -= 1
    return j


def place(path, section, viz):
    lines = path.read_text().split("\n") if path.exists() else ["# Overlay (widget placements)"]
    remove_viz(lines, viz["name"])
    if section == "_top":
        idx = next((i for i, l in enumerate(lines) if re.match(r"^viz:\s*$", l)), None)
        if idx is None:
            lines += ["viz:"] + item_lines(viz, 2)
        else:
            e = block_end(lines, idx, 0)
            lines[e:e] = item_lines(viz, 2)
    else:
        sec = next((i for i, l in enumerate(lines) if re.match(r"^sections:\s*$", l)), None)
        if sec is None:
            lines += ["sections:", f"  {yaml.safe_dump(section).strip().removesuffix(chr(10) + '...')}:", "    viz:"] + item_lines(viz, 6)
        else:
            send = block_end(lines, sec, 0)
            key_i = None
            for i in range(sec + 1, send):
                m = re.match(r"^  (\S.*?):\s*(?:\{\}|null)?\s*$", lines[i])
                if m and header_key(yaml.safe_load(m.group(1)) if m.group(1)[0] in "'\"" else m.group(1)) == header_key(section):
                    key_i = i
                    lines[i] = re.sub(r":\s*(?:\{\}|null)\s*$", ":", lines[i])
                    break
            if key_i is None:
                key = yaml.safe_dump(section, width=1000).strip().removesuffix("\n...").removesuffix("...").strip()
                lines[send:send] = [f"  {key}:", "    viz:"] + item_lines(viz, 6)
            else:
                kend = block_end(lines, key_i, 2)
                vz = next((i for i in range(key_i + 1, kend) if re.match(r"^    viz:\s*$", lines[i])), None)
                if vz is None:
                    lines[kend:kend] = ["    viz:"] + item_lines(viz, 6)
                else:
                    vend = block_end(lines, vz, 4)
                    lines[vend:vend] = item_lines(viz, 6)
    text = "\n".join(lines).rstrip("\n") + "\n"
    ov = yaml.safe_load(text)
    if section == "_top":
        ok = any(v.get("name") == viz["name"] for v in ov.get("viz") or [])
    else:
        ok = any(header_key(k) == header_key(section) and any(v.get("name") == viz["name"] for v in (s or {}).get("viz") or [])
                 for k, s in (ov.get("sections") or {}).items())
    if not ok:
        raise SystemExit(f"placement of {viz['name']} in {path} did not land")
    path.write_text(text)


def main():
    for p in yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()):
        place(HERE / "overlays" / f"{p['note']}.yml", p["section"], p["viz"])
        print(f"placed {p['viz']['name']} in {p['note']} / {p['section']}")


if __name__ == "__main__":
    main()

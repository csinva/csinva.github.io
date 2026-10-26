"""Screenshot a page of a built book (to check widgets, figures and text as a reader sees them).

    uv run python book/tools/shoot.py <build>/_build/html notes/stat/stat.html out.png [--find TEXT] [--dark]
        [--height 1000] [--width 1400] [--act '[{"viz":0,"input":1,"value":2}, {"viz":0,"button":0,"times":3},
                                                {"viz":0,"select":0,"value":"relu"}]']

Serves the build on a free local port, loads the page in headless Chrome inside a wrapper that sets the
light/dark theme, optionally moves widget controls (--act: index of the widget on the page, then which
input/select/button), scrolls to the first element containing --find, and saves a screenshot.
"""
import argparse, functools, http.server, pathlib, socketserver, subprocess, threading, urllib.parse, uuid, json
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WRAPPER = """<!doctype html><html><body style="margin:0"><iframe id="f" style="border:0"></iframe><script>
const q = new URLSearchParams(location.search), f = document.getElementById('f');
localStorage.setItem('theme', q.get('mode')); localStorage.setItem('mode', q.get('mode'));
f.style.height = q.get('h') + 'px'; f.style.width = q.get('w') + 'px'; f.src = q.get('page');
f.onload = () => setTimeout(() => { const d = f.contentDocument, acts = JSON.parse(q.get('act') || '[]');
  for (const a of acts) { const v = d.querySelectorAll('.nb-viz')[a.viz];
    if (a.input !== undefined) { const i = v.querySelectorAll('input')[a.input]; i.value = a.value; i.dispatchEvent(new Event('input')); }
    if (a.select !== undefined) { const s = v.querySelectorAll('select')[a.select]; s.value = a.value; s.dispatchEvent(new Event('change')); }
    if (a.button !== undefined) { const b = v.querySelectorAll('button')[a.button]; for (let k = 0; k < (a.times || 1); k++) b.click(); } }
  const t = q.get('text'); if (!t) return;
  const el = [...d.querySelectorAll('h1,h2,h3,h4,p,li,figcaption,.nb-viz,summary')].find(e => e.textContent.includes(t));
  if (el) f.contentWindow.scrollTo({top: el.getBoundingClientRect().top + f.contentWindow.scrollY - 70, behavior: 'instant'}); }, 9000);
</script></body></html>"""
ap = argparse.ArgumentParser()
ap.add_argument("build_html"); ap.add_argument("page"); ap.add_argument("out")
ap.add_argument("--find", default=""); ap.add_argument("--dark", action="store_true"); ap.add_argument("--height", type=int, default=1000)
ap.add_argument("--width", type=int, default=1400); ap.add_argument("--act", default="[]")
a = ap.parse_args()
root = pathlib.Path(a.build_html).resolve(); name = f"__shot{uuid.uuid4().hex[:8]}.html"
(root / name).write_text(WRAPPER)
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args): pass
with socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root))) as srv:
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    q = urllib.parse.urlencode({"page": a.page, "text": a.find, "mode": "dark" if a.dark else "light", "h": a.height, "w": a.width, "act": a.act})
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={a.width},{a.height}",
                    "--virtual-time-budget=20000", f"--screenshot={pathlib.Path(a.out).resolve()}", f"http://127.0.0.1:{srv.server_address[1]}/{name}?{q}"],
                   capture_output=True, timeout=180)
    srv.shutdown()
(root / name).unlink(missing_ok=True)
print(a.out)

"""Trich xuat khoi <script> tu INDEX_HTML trong web_server.cpp de kiem tra cu phap JS.
Dung:
    python tools/extract_web_js.py            -> output/_web_js_check.js
    node --check output/_web_js_check.js      -> bao loi cu phap (neu co)
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "firmware_esp32", "main", "web_server.cpp")
OUT = os.path.join(ROOT, "output", "_web_js_check.js")

src = open(SRC, encoding="utf-8", errors="ignore").read()
m = re.search(r'INDEX_HTML\[\]\s*=\s*R"rawliteral\((.*?)\)rawliteral"', src, re.S)
if not m:
    raise SystemExit("Khong tim thay INDEX_HTML trong web_server.cpp")

html = m.group(1)
s = html.find("<script>")
e = html.find("</script>")
if s < 0 or e < 0:
    raise SystemExit("Khong tim thay khoi <script>")

js = html[s + len("<script>"):e]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(js)
print(f"Da trich {len(js)} bytes JS -> {OUT}")

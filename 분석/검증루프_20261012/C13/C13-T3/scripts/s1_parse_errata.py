# ERRATA.txt E-* 항목 블록 추출 (읽기 전용)
import re, json, sys, io
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012"
raw = open(BASE + r"\ERRATA.txt", "rb").read()
try:
    txt = raw.decode("utf-8")
    enc = "utf-8"
except UnicodeDecodeError:
    txt = raw.decode("utf-8", errors="replace"); enc = "utf-8(replace)"
lines = txt.splitlines()
items = []
cur = None
section = None
hdr = re.compile(r"^\s*(E-C\d\d[A-Za-z0-9\-~]*)\s*(.*)$")
for i, l in enumerate(lines, 1):
    m = re.match(r"^=+\s*(C\d\d)|^==== \[?(C\d\d)|^==== (C\d\d)", l)
    if l.startswith("====") or l.startswith("================"):
        mm = re.search(r"C\d\d", l)
        if mm: section = mm.group(0)
        cur = None; continue
    if l.startswith("----") or l.startswith("반영 우선순위") or l.startswith("--- "):
        cur = None
        continue
    m = hdr.match(l)
    if m and not l.startswith("  E-"):
        cur = {"id": m.group(1), "line": i, "section": section, "text": l.strip()}
        items.append(cur)
    elif l.startswith("  E-"):
        # 재검 결과 줄(부가) — 별도 기록
        m2 = hdr.match(l)
        items.append({"id": m2.group(1), "line": i, "section": section, "text": l.strip(), "addendum": True})
        cur = None
    elif cur is not None and l.strip():
        cur["text"] += " / " + l.strip()
json.dump({"encoding": enc, "items": items}, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(enc, len(items))

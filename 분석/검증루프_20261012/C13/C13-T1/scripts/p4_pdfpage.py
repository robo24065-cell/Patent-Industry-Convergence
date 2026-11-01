# -*- coding: utf-8 -*-
# PDF 특정 쪽의 단어를 좌표 순(행 단위)으로 출력한다. 사용: python -I p4_pdfpage.py <pdf> <page1based> [...]
import sys, fitz
sys.stdout.reconfigure(encoding="utf-8")
doc = fitz.open(sys.argv[1])
for p in sys.argv[2:]:
    pg = doc[int(p) - 1]
    ws = pg.get_text("words")
    rows = {}
    for x0, y0, x1, y1, w, *_ in ws:
        rows.setdefault(round(y0 / 4), []).append((x0, w))
    print(f"=== page {p}")
    for k in sorted(rows):
        print(" ".join(w for x, w in sorted(rows[k])))

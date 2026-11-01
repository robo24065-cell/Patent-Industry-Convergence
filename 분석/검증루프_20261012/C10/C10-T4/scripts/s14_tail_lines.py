import sys, fitz
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
d = fitz.open(T + "\\" + sys.argv[1] + ".pdf")
for pn in [int(x) for x in sys.argv[2:]]:
    pg = d[pn - 1]
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"]).strip()
            if t: print(pn, round(l["bbox"][0]), round(l["bbox"][2]), round(l["bbox"][3]), t[:70])

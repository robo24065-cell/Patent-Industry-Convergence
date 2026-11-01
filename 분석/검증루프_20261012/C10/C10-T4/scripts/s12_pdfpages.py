# C10-T4 s12: 테스트 빌드 PDF 쪽수·쪽별 하단·마지막 쪽 내용 비교
import sys, os, json, fitz
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012\보고서_초안_v11.pdf"
names = sys.argv[1:]
out = {}
def info(p):
    d = fitz.open(p); res = []
    for pg in d:
        ph = pg.rect.height
        ls = [(l["bbox"][1], l["bbox"][3], "".join(s["text"] for s in l["spans"]).strip()) for b in pg.get_text("dict")["blocks"] for l in b.get("lines", [])]
        ls = [x for x in ls if x[2] and x[0] < ph * 0.93]
        res.append(dict(bottom=round(max(x[1] for x in ls), 1) if ls else None, n_lines=len(ls), last=ls[-1][2][:50] if ls else ""))
    return res
base = info(R)
out["v11"] = dict(pages=len(base), last_bottom=base[-1]["bottom"])
for n in names:
    r = info(os.path.join(T, n + ".pdf"))
    diffp = [i + 1 for i, (a, b) in enumerate(zip(base, r)) if a["bottom"] != b["bottom"] or a["last"] != b["last"]]
    out[n] = dict(pages=len(r), last_bottom=r[-1]["bottom"], last_page_lines=r[-1]["n_lines"], last_text=r[-1]["last"], pages_differing_from_v11=diffp)
    print(n, out[n])
print("v11", out["v11"])
json.dump(out, open(os.path.join(T, "pages_" + "_".join(names) + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

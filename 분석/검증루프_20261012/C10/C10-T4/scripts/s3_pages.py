# C10-T4 s3: v10·v11 PDF 쪽별 본문 하단·첫/끝 문구 비교, 마지막 쪽 줄 높이·여유 줄 수
import sys, json, fitz
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
out = {}
for v in ["v9", "v10", "v11"]:
    d = fitz.open(fr"{R}\보고서_초안_{v}.pdf"); pages = []
    for i, pg in enumerate(d):
        ph = pg.rect.height
        lines = []
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t: lines.append((l["bbox"][1], l["bbox"][3], t))
        body = [x for x in lines if x[0] < ph * 0.93]
        foot = [x for x in lines if x[0] >= ph * 0.93]
        pages.append(dict(p=i + 1, bottom=round(max(x[1] for x in body), 1) if body else None, ph=round(ph, 1), first=body[0][2][:30] if body else "",
                          last=body[-1][2][:40] if body else "", foot=[x[2] for x in foot][:2], foot_top=round(min(x[0] for x in foot), 1) if foot else None))
    last = d[-1]; ph = last.rect.height
    ls = sorted([(l["bbox"][1], l["bbox"][3]) for b in last.get_text("dict")["blocks"] for l in b.get("lines", []) if l["bbox"][1] < ph * 0.93])
    gaps = [round(ls[k + 1][0] - ls[k][0], 2) for k in range(len(ls) - 1) if 0 < ls[k + 1][0] - ls[k][0] < 30]
    out[v] = dict(n=len(pages), pages=pages, last_line_pitch_median=sorted(gaps)[len(gaps) // 2] if gaps else None)
json.dump(out, open(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\cache\s3_pages.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for v in out:
    print(v, out[v]["n"], "pitch", out[v]["last_line_pitch_median"])
    for p in out[v]["pages"]: print(" ", p["p"], p["bottom"], p["foot_top"], "|", p["first"], "…", p["last"])

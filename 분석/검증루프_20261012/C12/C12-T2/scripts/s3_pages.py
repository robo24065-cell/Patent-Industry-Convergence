# C12-T2 s3: v11/v12/v13 PDF 쪽별 본문 하단·마지막 쪽 여유(pt)·줄 간격
import sys, json, fitz, hashlib
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
OUT = sys.argv[1]
BODY_LIMIT = 785.2  # md2docx 하단 여백 20mm 기준(C10-T4 실측)
res = {}
for v in ["v11", "v12", "v13"]:
    p = R + "\보고서_초안_%s.pdf" % v
    d = fitz.open(p)
    pages = []
    for i in range(d.page_count):
        pg = d[i]; ph = pg.rect.height
        lines = []
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t: lines.append((round(l["bbox"][1], 1), round(l["bbox"][3], 1), t[:60]))
        lines.sort()
        body = [x for x in lines if x[0] < ph * 0.93]
        foot = [x for x in lines if x[0] >= ph * 0.93]
        pages.append(dict(page=i + 1, n_lines=len(body), first=body[0][2] if body else "", last=body[-1][2] if body else "",
                          body_bottom=body[-1][1] if body else 0, footer=[f[2] for f in foot]))
    last = pages[-1]
    # 마지막 쪽 줄 간격: 연속 줄 top 차이의 중앙값
    pg = d[d.page_count - 1]
    tops = sorted(set(round(l["bbox"][1], 1) for b in pg.get_text("dict")["blocks"] for l in b.get("lines", []) if "".join(s["text"] for s in l["spans"]).strip()))
    gaps = sorted(b - a for a, b in zip(tops, tops[1:]) if 5 < b - a < 30)
    lh = gaps[len(gaps) // 2] if gaps else None
    res[v] = dict(pdf_sha1=hashlib.sha1(open(p, "rb").read()).hexdigest(), pages=d.page_count, page_height=d[0].rect.height,
                  last_body_bottom=last["body_bottom"], slack_pt=round(BODY_LIMIT - last["body_bottom"], 1), line_gap_pt=lh,
                  slack_lines=round((BODY_LIMIT - last["body_bottom"]) / lh, 2) if lh else None, per_page=pages)
for v in res:
    r = res[v]; print(v, r["pages"], "last_bottom", r["last_body_bottom"], "slack", r["slack_pt"], "gap", r["line_gap_pt"], "slack_lines", r["slack_lines"])
# 쪽 배치 비교
for a, b in [("v11", "v12"), ("v12", "v13")]:
    diff = [(x["page"], x["first"][:25], y["first"][:25], x["body_bottom"], y["body_bottom"]) for x, y in zip(res[a]["per_page"], res[b]["per_page"]) if x["first"] != y["first"] or abs(x["body_bottom"] - y["body_bottom"]) > 0.5]
    res["cmp_%s_%s" % (a, b)] = diff
    print(a, b, "pages differing:", len(diff)); [print("  ", z) for z in diff]
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

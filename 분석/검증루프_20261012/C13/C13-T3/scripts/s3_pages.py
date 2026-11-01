# v13/v14 PDF 쪽별 본문 하단·마지막 쪽 여유 (C12-T2 s3 방식 재사용)
import sys, json, fitz, hashlib
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
BODY_LIMIT = 785.2
res = {}
for v in ["v13", "v14"]:
    p = R + "\보고서_초안_%s.pdf" % v
    d = fitz.open(p); pages = []
    for i in range(d.page_count):
        pg = d[i]; ph = pg.rect.height; lines = []
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t: lines.append((round(l["bbox"][1], 1), round(l["bbox"][3], 1), t[:50]))
        lines.sort(); body = [x for x in lines if x[0] < ph * 0.93]
        pages.append(dict(page=i + 1, first=body[0][2] if body else "", last=body[-1][2] if body else "", body_bottom=body[-1][1] if body else 0))
    pg = d[d.page_count - 1]
    tops = sorted(set(round(l["bbox"][1], 1) for b in pg.get_text("dict")["blocks"] for l in b.get("lines", []) if "".join(s["text"] for s in l["spans"]).strip()))
    gaps = sorted(b - a for a, b in zip(tops, tops[1:]) if 5 < b - a < 30); lh = gaps[len(gaps) // 2] if gaps else None
    res[v] = dict(pdf_sha1=hashlib.sha1(open(p, "rb").read()).hexdigest(), pages=d.page_count, last_body_bottom=pages[-1]["body_bottom"],
                  slack_pt=round(BODY_LIMIT - pages[-1]["body_bottom"], 1), line_gap_pt=lh, per_page=pages)
    print(v, res[v]["pages"], res[v]["last_body_bottom"], res[v]["slack_pt"], lh)
diff = [(x["page"], x["body_bottom"], y["body_bottom"], x["first"][:20], y["first"][:20]) for x, y in zip(res["v13"]["per_page"], res["v14"]["per_page"]) if x["first"] != y["first"] or abs(x["body_bottom"] - y["body_bottom"]) > 0.5]
res["cmp_v13_v14"] = diff; print(diff)
for p in res["v14"]["per_page"]: print(p["page"], p["body_bottom"], p["first"][:30], "|", p["last"][:30])
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)

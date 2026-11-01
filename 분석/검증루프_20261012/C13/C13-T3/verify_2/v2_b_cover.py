# verify_2 (b): 통합표 행 구성·범위 집계·미반영 후보 포괄성
import csv, json, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
T3 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T3"
st = json.load(open(T3 + r"\cache\errata_status.json", encoding="utf-8"))
rows = list(csv.DictReader(open(T3 + r"\errata_consolidated.csv", encoding="utf-8-sig")))
print("통합표 행:", len(rows), collections.Counter(r["scope"] for r in rows))
print("in_manuscript 분포:", collections.Counter(r["in_manuscript"].split("(")[0].split(":")[0] for r in rows))
body = [r for r in rows if r["scope"] == "body" and r["need"] != "제외"]
print("본문(제외 빼고):", len(body), " 그림:", sum(r["scope"] == "figure" for r in rows))
print("Δ글자 합: body", sum(int(r["delta_chars"] or 0) for r in body), " body+figure", sum(int(r["delta_chars"] or 0) for r in rows if r["scope"] in ("body", "figure")))
# 고유 상태
u = {}
for o in st:
    if o.get("addendum"): continue
    u.setdefault(o["id"], set()).add(o["status"])
print("고유 ID:", len(u), " 상태 충돌 ID:", [k for k, v in u.items() if len(v) > 1])
cu = collections.Counter(next(iter(v)).split(":")[0] for v in u.values())
print("고유 ID 상태:", cu)
cand = sorted(k for k, v in u.items() if "미반영후보" in v)
listed = {r["id"].split("(")[0] for r in rows}
print("미반영후보 고유", len(cand), " 통합표에 없음:", [k for k in cand if k not in listed])
print("통합표에 있으나 미반영후보 아님:", sorted((k, next(iter(u[k])) if k in u else "ERRATA에 없음") for k in listed if k not in cand))
# 원고 밖 58 의 접두 분포
out = [r for r in rows if r["scope"] == "outside"]
print("outside 행", len(out), collections.Counter((r["id"].split("-")[-1][:1]) for r in out))
for r in out: print("  ", r["id"], "|", r["note"][:70])

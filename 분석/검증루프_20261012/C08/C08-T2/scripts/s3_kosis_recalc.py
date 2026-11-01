# C07-T1 내려받은 KOSIS 원천(JSON)에서 원고 194·253행 값 직접 재계산 (읽기만)
import json, sys, math, collections
sys.stdout.reconfigure(encoding="utf-8")
p = sys.argv[1]
rows = json.load(open(p, encoding="utf-8"))
V = collections.defaultdict(dict)
for r in rows:
    if r.get("C1") != "00": continue
    try: v = float(r["DT"])
    except: v = None
    V[(r["C2"].lstrip("C"), r["ITM_ID"])][int(r["PRD_DE"])] = v
def g(code, itm, a=(2008, 2009), b=(2016, 2017)):
    d = V[(code, itm)]
    try:
        A = sum(d[y] for y in a) / 2; B = sum(d[y] for y in b) / 2
    except Exception: return None
    return round(100 * (math.log(B) - math.log(A)) / (b[0] - a[0]), 3)
out = {}
for code in ["26211", "26212", "26111", "29280"]:
    for itm, nm in [("T01", "사업체"), ("T02", "종사자"), ("T04", "출하")]:
        out[f"{code}_{nm}_0809_1617"] = g(code, itm)
        out[f"{code}_{nm}_0809_1516"] = g(code, itm, b=(2015, 2016))
out["31312_T01_years"] = {y: V[("31312", "T01")].get(y) for y in range(2007, 2014)}
out["26212_T01_years"] = {y: V[("26212", "T01")].get(y) for y in range(2007, 2021)}
out["26212_T04_years"] = {y: V[("26212", "T04")].get(y) for y in range(2014, 2021)}
out["26211_T04_years"] = {y: V[("26211", "T04")].get(y) for y in range(2014, 2021)}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)

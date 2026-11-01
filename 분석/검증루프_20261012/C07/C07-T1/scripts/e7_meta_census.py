import sys, json, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
tbl = "DT_1K52C03"
for typ in ("TBL", "PRD", "CMMT"):
    print(typ, json.dumps(klib.meta(tbl, typ, sub="meta_census"), ensure_ascii=False)[:800])
j = klib.meta(tbl, "ITM", sub="meta_census")
print(collections.Counter((r.get("OBJ_ID"), r.get("OBJ_NM")) for r in j))
for r in j:
    if r.get("OBJ_ID") not in ("A",) and not (r.get("OBJ_ID") == "B" and len(str(r.get("ITM_ID"))) > 3 and not str(r.get("ITM_ID")).endswith(("26211", "26212", "29280", "26111"))):
        print(r.get("OBJ_ID"), r.get("ITM_ID"), r.get("ITM_NM"), r.get("UP_ITM_ID"))

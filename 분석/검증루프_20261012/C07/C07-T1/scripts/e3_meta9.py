import sys, json, collections
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
for tbl in sys.argv[1:]:
    for typ in ("TBL", "PRD", "CMMT", "ITM"):
        j = klib.meta(tbl, typ)
        if typ != "ITM":
            print(tbl, typ, json.dumps(j, ensure_ascii=False)[:1200]); continue
        c = collections.Counter((r.get("OBJ_ID"), r.get("OBJ_NM")) for r in j)
        print(tbl, "ITM objs", c)
        for r in j:
            if r.get("OBJ_ID") == "ITEM" or r.get("OBJ_NM") in ("항목",):
                print("  item", r.get("ITM_ID"), r.get("ITM_NM"), r.get("UNIT_NM"))
        codes = [r for r in j if str(r.get("ITM_ID", "")).endswith(("26211", "26212", "26111", "29280", "31312"))]
        for r in codes[:10]:
            print("  code", r.get("OBJ_ID"), r.get("ITM_ID"), r.get("ITM_NM"))
        print("  sample", [ (r.get("OBJ_ID"), r.get("ITM_ID"), r.get("ITM_NM")) for r in j[:8]])

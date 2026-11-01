import sys, json
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
for tbl in ("DT_1FS1001", "DT_1FS1101"):
    for typ in ("TBL", "PRD", "CMMT", "NCD", "SOURCE"):
        try:
            j = klib.meta(tbl, typ)
            print(tbl, typ, json.dumps(j, ensure_ascii=False)[:1500])
        except klib.Stop as e:
            print("STOP", e); sys.exit()
        except Exception as e:
            print(tbl, typ, "ERR", e)

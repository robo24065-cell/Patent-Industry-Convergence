import sys, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
for typ in ("TBL", "PRD", "CMMT"):
    print(typ, json.dumps(klib.meta("DT_1F02001", typ, sub="meta_ipi"), ensure_ascii=False)[:1500])

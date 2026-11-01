import sys, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1\scripts")
import klib
pid = sys.argv[1]
j = klib.lst(pid)
for r in (j if isinstance(j, list) else [j]):
    print({k: r.get(k) for k in ("LIST_ID", "LIST_NM", "TBL_ID", "TBL_NM", "PRD_DE", "ORG_ID", "err", "errMsg") if r.get(k) is not None})

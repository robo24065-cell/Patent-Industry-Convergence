# -*- coding: utf-8 -*-
"""반박 검증 r3: 대안 A·B 캐시 base_rows의 해외 레코드 ksic가 pkr_alt의 k_reg/k_pure와 같은지, 원 base_rows 대비 KR 레코드 불변인지 독립 점검. 읽기만."""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.abspath(os.path.join(HERE, "..", ".."))
out = {}
for arm, col in [("t3_pkr_reg", "k_reg"), ("t3_pkr_pure", "k_pure")]:
    t = pq.read_table(os.path.join(W, "cache", arm, "base_rows.parquet"), columns=["ksic", "cc", "appl_no"])
    res = {}
    for cc in ["US", "JP", "EP", "CN"]:
        s = t.filter(pc.equal(t["cc"], cc)).to_pandas()
        a = pd.read_parquet(os.path.join(W, "cache", f"pkr_alt_{cc}.parquet"), columns=["appl_no", col])
        assert not a.appl_no.duplicated().any()
        m = s.merge(a, on="appl_no", how="left")
        res[cc] = dict(rows=len(s), matched=float(m[col].notna().mean()), same=float((m.ksic == m[col]).mean()))
        del s, a, m
    out[arm] = res
    print(arm, res, flush=True)
json.dump(out, open(os.path.join(HERE, "r3_cache_label_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

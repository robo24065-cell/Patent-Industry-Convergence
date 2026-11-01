# -*- coding: utf-8 -*-
"""K3 탐색(봉인 추출본 < 2023-01-01만): 국내 권리자 → 외국 권리자 이전 건수. 채택 판단 무관 기술통계."""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8")
o = pd.read_parquet(r"..\_ideas_R4\cache\reg_owner_lt2023.parquet")
assert o.d.max() < "20230101"
print("natl None 비율", o.ownerNatl.isna().mean().round(3))
print(o.cs.value_counts().head(12).to_dict())
print(o.ownerNatl.value_counts().head(12).to_dict())
o = o.sort_values(["rgst_no", "seq"])
o["prev_natl"] = o.groupby("rgst_no").ownerNatl.shift(1); o["prev_name"] = o.groupby("rgst_no").ownerName.shift(1)
o["yr"] = o.d.str[:4].astype(int)
KR = "대한민국"
tr = o[(o.seq > 0)]
print("이전 행", len(tr), "새 국적 결측", tr.ownerNatl.isna().mean().round(3), "이전 국적 결측", tr.prev_natl.isna().mean().round(3))
f = tr[(tr.prev_natl == KR) & tr.ownerNatl.notna() & (tr.ownerNatl != KR)]
print("KR→외국(국적 둘 다 기록)", len(f)); print(f.groupby("yr").rgst_no.nunique().to_dict())
print(f.ownerNatl.value_counts().head(10).to_dict()); print(f.cs.value_counts().head(8).to_dict())
g = tr[(tr.prev_natl.isna()) & tr.ownerNatl.notna() & (tr.ownerNatl != KR)]
print("이전 국적 결측→외국", len(g), g.groupby("yr").size().tail(12).to_dict())
print("국적 기록 연도별 비율", o.assign(h=o.ownerNatl.notna()).groupby("yr").h.mean().round(2).tail(18).to_dict())

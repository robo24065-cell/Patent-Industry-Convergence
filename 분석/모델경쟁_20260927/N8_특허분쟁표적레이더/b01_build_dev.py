# -*- coding: utf-8 -*-
"""dev 패널 생성(t0 2008–2020, 등록원부 사건 < 2023-01-01만). 파일럿 패널과 공통 열 일치 여부를 점검해 기록."""
import sys, json
sys.dont_write_bytecode = True
from n8lib import *

R = extract(SEAL); log("extract", {k: len(v) for k, v in R.items()})
X = prep(R); del R
FT = family_tokens(X["pt"]); log("fam", len(FT))
EXTU = external_unit(); KI = ksic_industry(); log("ext")
out = []
for t0 in range(TRAIN_MIN, 2021):
    P = build_t0(t0, X, FT, EXTU, KI)
    out.append(to32(P)); log(t0, len(P), int(P.y.sum()))
D = pd.concat(out, ignore_index=True)
D.to_parquet(f"{CACHE}/panel_dev.parquet", index=False)
log("saved", D.shape)
# 파일럿 패널과 비교(읽기 전용)
pl = pd.read_parquet(f"{MC}/_ideas_R5/cache/J2_panel.parquet")
m = D.merge(pl, on=["rgst_no", "t0"], how="outer", suffixes=("", "_pl"), indicator=True)
rep = {"rows_mine": len(D), "rows_pilot": len(pl), "merge": m._merge.value_counts().to_dict(),
       "pos_mine": int(D.y.sum()), "pos_pilot": int(pl.y.sum())}
both = m[m._merge == "both"]
diff = {}
for c in pl.columns:
    if c in ("rgst_no", "t0", "applNo", "ksic", "owner"):
        continue
    a, b = both[c].astype(float), both[c + "_pl"].astype(float)
    diff[c] = int((~((a - b).abs() < 1e-4) & ~(a.isna() & b.isna())).sum())
rep["n_mismatch_by_col"] = diff
json.dump(rep, open(f"{HERE}/b01_compare_pilot.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(rep)

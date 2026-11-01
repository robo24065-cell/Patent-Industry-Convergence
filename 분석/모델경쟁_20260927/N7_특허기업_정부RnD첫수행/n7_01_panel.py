# -*- coding: utf-8 -*-
"""dev 패널 생성(t0 2015–2020) + 파일럿 I7_panel과 대조. NTIS 과제연도 ≤ 2022만 읽는다."""
import os, numpy as np, pandas as pd
from n7lib import *

D = build_panel("dev")
assert D.t0.max() <= 2020 and (D.t0 + 2).max() < F_YEAR
D.to_parquet(os.path.join(CACHE, "dev_panel.parquet"), index=False)
log("saved", D.shape)
# 표본 표
tab = D.groupby("t0").agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean"), pos1=("y1", "sum"), nps=("nps_present", "mean"),
                          part=("g_part_any", "mean"), clusters=("bizno", "nunique")).reset_index()
print(tab.round(4).to_string(index=False))
tab.to_csv(os.path.join(OUT, "sample_dev.csv"), index=False, encoding="utf-8-sig")
# 파일럿 대조
P = pd.read_parquet(os.path.join(MC, r"_ideas_R4\cache\I7_panel.parquet"))
m = P.merge(D, on=["applicantCd", "t0"], how="outer", suffixes=("_p", "_n"), indicator=True)
print(m._merge.value_counts())
b = m[m._merge == "both"]
for c in ["y"] + M0 + M1_ADD + M2_ADD + M3_ADD:
    a1, a2 = b[c + "_p"].astype(float), b[c + "_n"].astype(float)
    same = ((a1 - a2).abs() < 1e-9) | (a1.isna() & a2.isna())
    print(f"{c:16s} 일치율 {same.mean():.5f}")

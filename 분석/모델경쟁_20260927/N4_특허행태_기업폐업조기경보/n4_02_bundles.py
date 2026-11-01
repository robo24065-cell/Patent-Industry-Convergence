# -*- coding: utf-8 -*-
"""dev 도전자 묶음 피처 → cache/feat_dev.parquet (base_dev와 같은 행 순서)."""
import os
import pandas as pd, numpy as np
from n4lib import *
from n4feat import *

MODE = "dev"
D = pd.read_parquet(os.path.join(CACHE, "base_dev.parquet"))
assert D.t0.max() + 2 < F_YEAR
AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
ns = load_nts()
ctx = load_ctx(MODE, set(D.applicantCd)); log("ctx")
out = []
for t0, B in D.groupby("t0"):
    X = build_all(B, t0, ctx, AF, ns)
    out.append(X); log(t0, X.shape)
X = pd.concat(out).reindex(D.index)
X.to_parquet(os.path.join(CACHE, "feat_dev.parquet"))
# 커버리지·레이블별 평균(해석용)
S = pd.concat([D[["t0", "y"]], X], axis=1)
desc = S.groupby("y").agg(lambda s: s.notna().mean()).T.add_prefix("nonnull_y").join(S.groupby("y").mean().T.add_prefix("mean_y"))
print(desc.round(3).to_string())
desc.to_csv(os.path.join(OUT, "feat_coverage_dev.csv"), encoding="utf-8-sig")

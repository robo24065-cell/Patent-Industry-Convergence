# -*- coding: utf-8 -*-
"""dev 패널(t0 2015–2020)에 도전자 묶음 피처를 붙여 cache/feat_dev.parquet로 저장 + 피처 결측·평균 표."""
import os, sys, numpy as np, pandas as pd
from n7lib import *
from n7feat import load_ctx, build_features, BUNDLES

D = pd.read_parquet(os.path.join(CACHE, "dev_panel.parquet"))
assert D.t0.max() <= 2020
ctx = load_ctx("dev")
log("ctx")
which = sys.argv[1:] or None
X = build_features(D, ctx, which)
fp = os.path.join(CACHE, "feat_dev.parquet")
if which and os.path.exists(fp):
    old = pd.read_parquet(fp)
    for c in X.columns:
        old[c] = X[c].values
    X = old
X.to_parquet(fp, index=False)
cov = pd.concat([D[["t0"]], X], axis=1).groupby("t0").agg(lambda s: s.notna().mean()).T.round(3)
mean = pd.concat([D[["t0", "y"]], X], axis=1)
print(cov.to_string())
cov.to_csv(os.path.join(OUT, "feat_coverage_dev.csv"), encoding="utf-8-sig")
log("done", X.shape)

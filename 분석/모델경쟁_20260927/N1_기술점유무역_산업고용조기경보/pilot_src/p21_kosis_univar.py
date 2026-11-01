# -*- coding: utf-8 -*-
"""아이디어 B 재정식화 2: 적합 없이(과적합 제거) 단일 신호의 원점별 Spearman ρ — 3년 상대성장(종사자·출하·부가가치). 원점 2008..2018(결과 ≤2021)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_kosis3y.parquet")
S = ["mom_KR", "mom_ALL", "dsh_KR", "frgn_mom", "mom_CN", "lvl_KR", "gap_CN_KR", "sh_KR", "g3_shp", "g3_emp", "g1_shp"]
out = []
for tg in ["y_emp", "y_shp", "y_va"]:
    for s in S:
        r = [spearmanr(g[tg], g[s], nan_policy="omit")[0] for t, g in P.groupby("t0") if g[s].notna().sum() > 20]
        r = np.array(r); out.append((tg, s, len(r), r.mean(), (r > 0).mean(), r.mean()/(r.std(ddof=1)/np.sqrt(len(r)))))
O = pd.DataFrame(out, columns=["target", "signal", "n_orig", "rho_mean", "pos_share", "t"])
print(O.round(3).to_string()); O.to_csv(f"{HERE}/p21_univar.csv", index=False)

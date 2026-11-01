# -*- coding: utf-8 -*-
"""아이디어 C 재정식화 2: KSIC 5자리 단위(48개)로 집계한 대중 수입점유율 3년 변화의 원점 내 순위 — 단일 신호 Spearman ρ. 결과 ≤2021."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ilib import *
pv = load_pat()
c = pd.read_csv(os.path.join(BASE, "분석", "fusion", "cn_import_yearly_ksic5.csv"), dtype={"ksic": str})
c = c[(c.year <= DEV_MAX) & (c.ksic_level == 5)].pivot_table(index="ksic", columns="year", values="cn_share_imp")
out = []
for t0 in range(2006, 2019):
    f = pat_feats(pv, t0).join(app_feats(t0))
    f["y"] = c[t0+3] - c[t0]; f["past"] = c[t0] - c[t0-3]; f["lvl"] = c[t0]
    f = f.dropna(subset=["y"])
    for s in ["mom_CN", "dsh_CN", "sh_CN", "gap_CN_KR", "newmom_CN", "apmom_CN", "past", "lvl"]:
        out.append((t0, s, spearmanr(f.y, f[s])[0]))
O = pd.DataFrame(out, columns=["t0", "sig", "rho"]).pivot(index="sig", columns="t0", values="rho")
O["mean"] = O.mean(1); O["pos"] = (O.drop(columns="mean") > 0).mean(1); print(O.round(2).to_string())

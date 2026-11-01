# -*- coding: utf-8 -*-
"""아이디어 B 재정식화 3: 원점 내 순위 변환(이상치 제거) + Ridge. 타깃 = 3년 상대성장의 원점 내 백분위.
지표 = 원점별 Spearman ρ(예측 vs 실제 3년 성장). skill = ρ_M − ρ_B (B=자기 이력만), 보조: 상위 1/3 성장 산업 AUC.
평가 v=2011..2018(결과 ≤2021), 학습 t0 ≤ v−3."""
import numpy as np, pandas as pd, sys
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_kosis3y.parquet")
def rk(g, cols):
    g = g.copy()
    for c in cols: g[c] = g[c].rank(pct=True).fillna(0.5)
    return g
PFS = {"pat_min": ["mom_KR", "dsh_KR"], "pat_all": ["mom_KR", "dsh_KR", "mom_ALL", "frgn_mom", "mom_CN", "lvl_KR", "sh_KR"]}
def run(tg, BF, PF, alpha=5.0, label=""):
    cols = BF + PF + [f"y_{tg}"]
    Q = pd.concat([rk(g, cols) for _, g in P.dropna(subset=[f"y_{tg}"]).groupby("t0")])
    res = []
    for v in range(2011, 2019):
        if v not in Q.t0.values: continue
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
        yraw = P[(P.t0 == v)].set_index("ksic").loc[te.ksic, f"y_{tg}"].values
        r = {"v": v}
        for lab, F in [("B", BF), ("M", BF+PF)]:
            if len(F) == 0: p = np.zeros(len(te))
            else: p = Ridge(alpha=alpha).fit(tr[F], tr[f"y_{tg}"]).predict(te[F])
            r[f"rho_{lab}"] = spearmanr(yraw, p)[0] if np.std(p) > 0 else 0.0
            top = (te[f"y_{tg}"] >= 2/3).astype(int)
            r[f"auc_{lab}"] = roc_auc_score(top, p) if np.std(p) > 0 else 0.5
        res.append(r)
    R = pd.DataFrame(res); d = R.rho_M - R.rho_B
    print(f"{label:28s} [{tg}] ρ_B {R.rho_B.mean():.3f}  ρ_M {R.rho_M.mean():.3f}  Δρ {d.mean():+.3f} 승률 {(d>0).mean():.2f} | AUC_B {R.auc_B.mean():.3f} AUC_M {R.auc_M.mean():.3f}")
    return R
for tg in ["emp", "shp", "va"]:
    BF = [f"g3_{tg}", f"g1_{tg}", f"lv_{tg}", "prod"]
    run(tg, [], PFS["pat_min"], label="0 vs 특허2")
    run(tg, BF, PFS["pat_min"], label="자기이력 vs +특허2")
    run(tg, BF, PFS["pat_all"], label="자기이력 vs +특허7")
R = run("emp", ["g3_emp", "g1_emp", "lv_emp", "prod"], PFS["pat_min"], label="(표) emp +특허2"); print(R.round(3).to_string())

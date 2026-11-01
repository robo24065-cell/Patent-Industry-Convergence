# -*- coding: utf-8 -*-
"""아이디어 B(특허→3년 고용 순위) 확장: 출원인 동학·무역(수출·대중 수입점유) 결합 실험. 순위 Ridge, 평가 v=2011..2018."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_kosis3y.parquet")
ext = []
for t0 in sorted(P.t0.unique()):
    a = app_feats(t0).join(trade_feats(t0), how="outer"); a["t0"] = t0; ext.append(a.reset_index())
E = pd.concat(ext); P = P.merge(E, on=["ksic", "t0"], how="left")
P.to_parquet(f"{CACHE}/panel_kosis3y_ext.parquet")
print("무역 결합률(xg3 비결측)", round(P.xg3.notna().mean(), 3), "cnsh", round(P.cnsh.notna().mean(), 3))
def rk(g, cols):
    g = g.copy()
    for c in cols: g[c] = g[c].rank(pct=True).fillna(0.5)
    return g
def run(tg, F, alpha=5.0, vs=range(2011, 2019)):
    Q = pd.concat([rk(g, F+[f"y_{tg}"]) for _, g in P.dropna(subset=[f"y_{tg}"]).groupby("t0")])
    out = []
    for v in vs:
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
        yraw = P[P.t0 == v].set_index("ksic").loc[te.ksic, f"y_{tg}"].values
        p = Ridge(alpha=alpha).fit(tr[F], tr[f"y_{tg}"]).predict(te[F]) if F else np.zeros(len(te))
        out.append((v, spearmanr(yraw, p)[0] if np.std(p) > 0 else 0.0, roc_auc_score((te[f"y_{tg}"] >= 2/3).astype(int), p) if np.std(p) > 0 else .5))
    return pd.DataFrame(out, columns=["v", "rho", "auc"])
base = ["mom_KR", "dsh_KR"]
exps = {"E0 특허2(챔피언)": base,
        "E1 +출원인 수 모멘텀": base+["apmom_KR"],
        "E2 +신규출원인 모멘텀": base+["newmom_KR"],
        "E3 +신규출원인 비중": base+["newsh_KR"],
        "E4 +수출 3년성장": base+["xg3"],
        "E5 +대중수입점유 변화": base+["dcnsh"],
        "E6 +대중수입점유 수준": base+["cnsh"],
        "E7 +해외모멘텀": base+["frgn_mom"],
        "E8 +CN 신규출원인": base+["newmom_CN"],
        "E9 +자기 고용추세": base+["g3_emp"]}
for tg in ["emp", "shp"]:
    ch = run(tg, exps["E0 특허2(챔피언)"])
    print(f"--- [{tg}] 챔피언 ρ {ch.rho.mean():.3f} AUC {ch.auc.mean():.3f}")
    for k, F in exps.items():
        R = run(tg, F); d = R.rho - ch.rho
        print(f"{k:22s} ρ {R.rho.mean():.3f}  Δρ {d.mean():+.3f} 승률 {(d>0).mean():.2f}  AUC {R.auc.mean():.3f}")

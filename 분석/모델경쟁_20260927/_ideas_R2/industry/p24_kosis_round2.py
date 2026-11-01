# -*- coding: utf-8 -*-
"""아이디어 B 2차 챔피언/도전자: 챔피언 = 특허2 + 대중수입점유 수준(cnsh). 타깃 종사자 3년 상대성장."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_kosis3y_ext.parquet")
pv = load_pat()
# 시차 2 특허(보수적 공개지연) 변형
L2 = []
for t0 in sorted(P.t0.unique()):
    f = pat_feats(pv, t0, lag=2)[["mom_KR", "dsh_KR"]].add_suffix("_L2"); f["t0"] = t0; L2.append(f.reset_index())
P = P.merge(pd.concat(L2), on=["ksic", "t0"], how="left")
def rk(g, cols):
    g = g.copy()
    for c in cols: g[c] = g[c].rank(pct=True).fillna(0.5)
    return g
def run(tg, F, alpha=5.0):
    Q = pd.concat([rk(g, F+[f"y_{tg}"]) for _, g in P.dropna(subset=[f"y_{tg}"]).groupby("t0")])
    out, coefs = [], []
    for v in range(2011, 2019):
        tr, te = Q[Q.t0 <= v-3], Q[Q.t0 == v]
        yraw = P[P.t0 == v].set_index("ksic").loc[te.ksic, f"y_{tg}"].values
        m = Ridge(alpha=alpha).fit(tr[F], tr[f"y_{tg}"]); p = m.predict(te[F]); coefs.append(m.coef_)
        out.append((v, spearmanr(yraw, p)[0], roc_auc_score((te[f"y_{tg}"] >= 2/3).astype(int), p)))
    return pd.DataFrame(out, columns=["v", "rho", "auc"]), np.mean(coefs, 0)
CH = ["mom_KR", "dsh_KR", "cnsh"]
ch, cf = run("emp", CH); print(f"챔피언 ρ {ch.rho.mean():.3f} AUC {ch.auc.mean():.3f} 계수 {dict(zip(CH, cf.round(3)))}")
print(ch.round(3).to_string())
exps = {"A1 특허 시차2로 교체": ["mom_KR_L2", "dsh_KR_L2", "cnsh"],
        "A2 +dcnsh": CH+["dcnsh"], "A3 +sh_KR": CH+["sh_KR"], "A4 +lvl_KR(규모)": CH+["lvl_KR"],
        "A5 +lv_emp(고용규모)": CH+["lv_emp"], "A6 +prod(노동생산성)": CH+["prod"],
        "A7 +mom_CN": CH+["mom_CN"], "A8 +gap_CN_KR": CH+["gap_CN_KR"], "A9 -dsh_KR": ["mom_KR", "cnsh"],
        "A10 -mom_KR": ["dsh_KR", "cnsh"], "A11 alpha=1": None, "A12 alpha=20": None}
for k, F in exps.items():
    if F is None: R, _ = run("emp", CH, alpha=float(k.split("=")[1]))
    else: R, _ = run("emp", F)
    d = R.rho - ch.rho
    print(f"{k:22s} ρ {R.rho.mean():.3f}  Δρ {d.mean():+.3f} 승률 {(d>0).mean():.2f}  AUC {R.auc.mean():.3f}")
# 적합 없는 합성점수(사전 지정 부호): rank(mom_KR)+rank(dsh_KR)-rank(cnsh), 전 원점
r = []
for t, g in P.dropna(subset=["y_emp"]).groupby("t0"):
    s = g.mom_KR.rank(pct=True)+g.dsh_KR.rank(pct=True)-g.cnsh.rank(pct=True).fillna(.5)
    r.append((t, spearmanr(g.y_emp, s)[0]))
r = pd.DataFrame(r, columns=["t0", "rho"]); print("무적합 합성 ρ(원점별)", r.round(3).values.tolist(), "평균", round(r.rho.mean(), 3))

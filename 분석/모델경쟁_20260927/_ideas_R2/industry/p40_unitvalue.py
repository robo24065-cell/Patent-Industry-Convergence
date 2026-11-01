# -*- coding: utf-8 -*-
"""아이디어 C' 파일럿: '기술 고도화 → 수출 단가(품질) 사다리 상승'.
단위 = (HS6, 국가 c ∈ 5개국, t0). 상대단가 r = log(단가_c) − log(나머지 4개국 단가 중앙값), 단가 = 수출액/순중량(kg).
y = r(t0+3) − r(t0), (c,t0) 내 중앙값 차감. 특허 = KSIC(h)의 국가 c 특허 점유 변화·특화도·모멘텀(출원 ≤ t0−1).
기준선 = 무역만(과거 3년 상대단가 변화, 상대단가 수준, 수출 규모, 점유). 결과 ≤2021 → t0 ≤ 2018."""
import sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
STEP = 3
X = pd.read_csv(os.path.join(B, "_shared", "ext", "R2ind_comtrade_exports_5c_hs6_2004_2021.csv"), dtype={"hs6": str})
X = X[(X.year <= DEV_MAX)]; X["hs6"] = X.hs6.str.zfill(6)
X["c"] = X.reporter.map({410: "KR", 156: "CN", 392: "JP", 842: "US", 276: "EP"})
X = X[(X.net_wgt_kg > 0) & (X.export_usd >= 1e5)]
X["luv"] = np.log(X.export_usd / X.net_wgt_kg)
UV = X.pivot_table(index=["hs6", "year"], columns="c", values="luv"); XV = X.pivot_table(index=["hs6", "year"], columns="c", values="export_usd")
yrs = sorted(X.year.unique()); mp = load_map(); P30 = pd.read_parquet(f"{CACHE}/panel_spec_step3.parquet")[["hs6", "c", "t0", "p_dls", "p_rta", "p_mom", "p_lvl", "b_sh", "ksic"]]
CC = ["KR", "CN", "JP", "US", "EP"]
def rel(y):
    U = UV.xs(y, level="year").reindex(columns=CC); out = {}
    for c in CC: out[c] = U[c] - U.drop(columns=c).median(1, skipna=True).where(U.drop(columns=c).notna().sum(1) >= 2)
    return pd.DataFrame(out)
rows = []
for t0 in yrs:
    if t0+STEP not in yrs or t0 > DEV_MAX-STEP: continue
    R0, R3 = rel(t0), rel(t0+STEP); Rm = rel(t0-STEP) if t0-STEP in yrs else None
    for c in CC:
        d = pd.DataFrame({"r0": R0[c], "r3": R3[c]}).dropna()
        d["y"] = d.r3 - d.r0; d["b_rpast"] = (d.r0 - Rm[c].reindex(d.index)) if Rm is not None else np.nan
        d["b_r0"] = d.r0; d["c"] = c; d["t0"] = t0; rows.append(d.reset_index().rename(columns={"index": "hs6"}))
P = pd.concat(rows, ignore_index=True).merge(P30, on=["hs6", "c", "t0"], how="inner")
for col in ["y", "b_rpast", "b_r0"]: P[col] = P[col] - P.groupby(["c", "t0"])[col].transform("median")
P["y"] = P.y.clip(-2, 2); P.to_parquet(f"{CACHE}/panel_uv.parquet")
print("행", len(P), "원점", sorted(P.t0.unique()), P.groupby("c").size().to_dict(), "y sd", round(P.y.std(), 3))
BF = ["b_rpast", "b_r0", "b_sh"]; PF = ["p_dls", "p_rta", "p_mom", "p_lvl"]; res = []
for v in sorted(P.t0.unique()):
    tr, te = P[P.t0 <= v-STEP], P[P.t0 == v]
    if len(tr) < 500: continue
    r = {"v": v, "n": len(te), "zero": mae(te.y, 0)}
    for lab, F in [("B", BF), ("M", BF+PF)]:
        p = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             objective="reg:absoluteerror", n_jobs=2, random_state=0).fit(tr[F], tr.y).predict(te[F])
        r[lab] = mae(te.y, p); te2 = te.assign(p=p)
        r[f"rho_{lab}"] = np.mean([spearmanr(g.y, g.p)[0] for _, g in te2.groupby("c")])
    res.append(r)
R = pd.DataFrame(res); print(R.round(4).to_string())
print(f"무역만 skill vs 0 {(1-R.B/R.zero).mean():.4f} | 특허 증분 skill {(1-R.M/R.B).mean():.4f} 승률 {(R.M<R.B).mean():.2f} | ρ_B {R.rho_B.mean():.3f} ρ_M {R.rho_M.mean():.3f}")
# 단일 신호 ρ (적합 없음)
for s in PF:
    print(s, round(np.mean([spearmanr(g.y, g[s], nan_policy='omit')[0] for _, g in P.groupby(["c", "t0"])]), 4))

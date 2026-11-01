# -*- coding: utf-8 -*-
"""G2 파일럿 — '기술 국산화 → 무역 특화 전환' : HS6 품목 무역특화지수 TSI=(X−M)/(X+M) 의 다음 3년 변화 예측.
TSI는 수출·수입 비율이라 세계 경기(공통 충격)가 상당 부분 상쇄된다.
단위 HS6 × 원점 t0(2005..2018). 타깃 y = TSI(t0+1..t0+3 합) − TSI(t0−2..t0 합).
개발 원점 v=2010..2018, 학습 원점 ≤ v−3. 🔒 최종시험 t0=2021 → 2022–24 (F=2022). 무역 2022+ 로드·수집 안 함.
B0 = 0(변화 없음), B1 = 품목 자기이력 XGB, M1 = B1 + KSIC 특허 피처.
"""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

m = load_map(); imp = load_imports(); ex = load_exports()
M = trade_panel(imp, "WLD"); X = ex.pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")
idx = M.index.intersection(X.index); M = M.loc[idx]; X = X.loc[idx].reindex(columns=M.columns)
pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
hk = m[m.hs6.isin(idx)]
print("HS6 교집합", len(idx))
rows = []
for t0 in range(2005, 2019):
    A_, B_, F_ = [t0 - 2, t0 - 1, t0], [t0 - 5, t0 - 4, t0 - 3], [t0 + 1, t0 + 2, t0 + 3]
    yrs = B_ + A_ + F_
    ok = (M[yrs].notna() & X[yrs].notna()).all(axis=1) & ((M[yrs].fillna(0) + X[yrs].fillna(0)) > 0).all(axis=1)
    mm, xx = M.loc[ok, yrs].fillna(0), X.loc[ok, yrs].fillna(0)
    tsi = lambda c: (xx[c].sum(1) - mm[c].sum(1)) / (xx[c].sum(1) + mm[c].sum(1))
    tA, tB, tF = tsi(A_), tsi(B_), tsi(F_)
    df = pd.DataFrame({"hs6": mm.index, "t0": t0, "tsi": tA.values, "tsi_chg": (tA - tB).values,
                       "lx": np.log1p(xx[A_].sum(1)).values, "lm": np.log1p(mm[A_].sum(1)).values,
                       "xmom": np.log((xx[A_].sum(1) + 1) / (xx[B_].sum(1) + 1)).values,
                       "mmom": np.log((mm[A_].sum(1) + 1) / (mm[B_].sum(1) + 1)).values,
                       "tsi1": ((xx[t0] - mm[t0]) / (xx[t0] + mm[t0] + 1e-9)).values})
    df["y"] = (tF - tA).values
    pf = patent_feats(t0, pan)
    ff = hk.merge(pf, on="k5").drop(columns="k5").groupby("hs6").mean()
    rows.append(df.merge(ff, left_on="hs6", right_index=True, how="left"))
D = pd.concat(rows, ignore_index=True)
D.to_parquet(os.path.join(CACHE, "g2_tsi_panel.parquet"), index=False)
print("표본", D.groupby("t0").size().to_dict())
OWN = ["tsi", "tsi_chg", "lx", "lm", "xmom", "mmom", "tsi1"]
PAT = [c for c in D.columns if c.startswith(("pm_", "pl_", "p_"))]
CORE = ["pm_KR", "pm_FOR", "p_gap_for_kr", "p_domshare", "p_domshare_chg", "pm_KRin", "pm_KRout", "p_outshare", "p_outshare_chg", "p_krshare5", "pm_CN", "pm_CNout"]
SETS = {"B1": OWN, "M1_allpat": OWN + PAT, "M2_core": OWN + CORE}


def fit(tr, te, cols):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=10, objective="reg:absoluteerror", nthread=2, random_state=0)
    mdl.fit(tr[cols], tr.y); return mdl.predict(te[cols])


res = []
for v in range(2010, 2019):
    tr, te = D[D.t0 <= v - 3], D[D.t0 == v]
    up = (te.y >= te.y.quantile(0.8)).astype(int)
    r = {"v": v, "n": len(te), "mae_B0": np.abs(te.y).mean()}
    for k, cols in SETS.items():
        p = fit(tr, te, cols)
        r["mae_" + k] = np.abs(te.y - p).mean(); r["rho_" + k] = spearman(p, te.y); r["auc_" + k] = roc_auc_score(up, p)
    res.append(r)
R = pd.DataFrame(res)
s = {"mae_B0": round(R.mae_B0.mean(), 4)}
for k in SETS:
    s[k] = {"mae": round(R["mae_" + k].mean(), 4), "skill_vsB0": round(1 - R["mae_" + k].mean() / R.mae_B0.mean(), 4),
            "skill_vsB1": round(float((1 - R["mae_" + k] / R.mae_B1).mean()), 4), "win_vsB1": round(float((R["mae_" + k] < R.mae_B1).mean()), 2),
            "rho": round(R["rho_" + k].mean(), 4), "auc_up20": round(R["auc_" + k].mean(), 4)}
print(json.dumps(s, ensure_ascii=False, indent=1))
R.to_csv(os.path.join(HERE, "g2_tsi_result.csv"), index=False, encoding="utf-8-sig")

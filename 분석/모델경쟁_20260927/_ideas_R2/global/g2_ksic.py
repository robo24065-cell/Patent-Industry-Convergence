# -*- coding: utf-8 -*-
"""G2 4차 — KSIC5 산업 단위(48개) 수출 3년 상대성장. 특허 피처와 같은 해상도에서 기여 확인.
균등배분 집계, 창 전체 관측 품목만. 개발 원점 v=2010..2018, 학습 ≤ v−3. 🔒 2022+ 미사용."""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
m = load_map(); ex = load_exports(); imp = load_imports()
X = ex.pivot_table(index="hs6", columns="year", values="value", aggfunc="sum"); M = trade_panel(imp, "WLD").reindex(X.index)
hk = m[m.hs6.isin(X.index)]; pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
rows = []
for t0 in range(2005, 2019):
    A_, B_, F_ = [t0-2, t0-1, t0], [t0-5, t0-4, t0-3], [t0+1, t0+2, t0+3]
    s = X[B_ + A_ + F_]; s = s[(s > 0).all(axis=1)]
    t = s.reset_index().merge(hk, on="hs6"); w = t.groupby("hs6").k5.transform("count")
    for c in B_ + A_ + F_: t[c] = t[c] / w
    g = t.groupby("k5")[B_ + A_ + F_].sum()
    mm = M.loc[s.index, B_ + A_].fillna(0).reset_index().merge(hk, on="hs6")
    for c in B_ + A_: mm[c] = mm[c] / mm.groupby("hs6").k5.transform("count")
    mm = mm.groupby("k5")[B_ + A_].sum().reindex(g.index)
    A, B, Fw = g[A_].sum(1), g[B_].sum(1), g[F_].sum(1)
    df = pd.DataFrame({"k5": g.index, "t0": t0, "lvl": np.log(A.values), "mom3": np.log(A / B).values,
                       "mom1": np.log(g[t0] / g[t0-1]).values, "imp_ratio": np.log((mm[A_].sum(1) + 1) / A).values,
                       "y_raw": np.log(Fw / A).values})
    rows.append(df.merge(patent_feats(t0, pan), on="k5", how="left"))
D = pd.concat(rows, ignore_index=True); D["y"] = D.y_raw - D.groupby("t0").y_raw.transform("median")
OWN = ["lvl", "mom3", "mom1", "imp_ratio"]
LVL = [c for c in D.columns if c.startswith("pl_")] + ["p_domshare", "p_outshare", "p_krshare5"]
PAT = [c for c in D.columns if c.startswith(("pm_", "pl_", "p_"))]
def xg(tr, te, c):
    mdl = xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=5, objective="reg:absoluteerror", nthread=2, random_state=0)
    mdl.fit(tr[c], tr.y); return mdl.predict(te[c])
def rg(tr, te, c):
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 15))); mdl.fit(tr[c], tr.y); return mdl.predict(te[c])
SETS = {"B1": OWN, "B1+out": OWN + ["p_outshare"], "B1+LVL": OWN + LVL, "B1+PAT": OWN + PAT}
res = []
for v in range(2010, 2019):
    tr, te = D[D.t0 <= v-3], D[D.t0 == v]; r = {"v": v, "B0": np.abs(te.y).mean()}
    for fn, f in [("xgb", xg), ("ridge", rg)]:
        for k, c in SETS.items():
            p = f(tr, te, c); r[f"{fn}_{k}"] = np.abs(te.y - p).mean(); r[f"rho_{fn}_{k}"] = spearman(p, te.y)
    res.append(r)
R = pd.DataFrame(res); out = {}
for fn in ["xgb", "ridge"]:
    for k in SETS:
        c = f"{fn}_{k}"
        out[c] = {"skill_vsB0": round(1 - R[c].mean() / R.B0.mean(), 4), "win_vsB0": round(float((R[c] < R.B0).mean()), 2),
                  "skill_vsB1": round(float((1 - R[c] / R[f"{fn}_B1"]).mean()), 4), "win_vsB1": round(float((R[c] < R[f"{fn}_B1"]).mean()), 2),
                  "rho": round(R["rho_" + c].mean(), 3)}
print(pd.DataFrame(out).T.to_string()); R.to_csv(os.path.join(HERE, "g2_ksic_result.csv"), index=False, encoding="utf-8-sig")

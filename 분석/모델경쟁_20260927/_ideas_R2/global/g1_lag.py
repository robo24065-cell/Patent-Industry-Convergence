# -*- coding: utf-8 -*-
"""G1 4차 — 결과창은 3년 그대로(최종시험 t0=2021 유지), 특허 피처를 과거 시차(t0−2, t0−4 시점 계산)로 추가.
5년 창에서 특허 기여가 커진 결과(+0.020)가 '긴 선행 시차' 때문인지 확인. 개발 원점 v=2010..2018.
"""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

D = pd.read_parquet(os.path.join(CACHE, "g1_panel_M.parquet"))
m = load_map(); pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
hk = m[m.hs6.isin(D.hs6.unique())]
PATC = [c for c in D.columns if c.startswith(("pm_", "pl_", "p_"))]
adds = []
for L in [2, 4]:
    parts = []
    for t0 in sorted(D.t0.unique()):
        pf = patent_feats(t0 - L, pan)
        ff = hk.merge(pf, on="k5").drop(columns="k5").groupby("hs6").mean().add_suffix(f"_L{L}")
        ff["t0"] = t0; parts.append(ff.reset_index())
    lag = pd.concat(parts)
    D = D.merge(lag, on=["hs6", "t0"], how="left")
OWN = ["lvl", "mom3", "mom1", "vol", "cn_sh", "cn_sh_chg", "k_mom3"]
L2 = [c + "_L2" for c in PATC]; L4 = [c + "_L4" for c in PATC]
SETS = {"B1": OWN, "M1": OWN + PATC, "M1+L2": OWN + PATC + L2, "M1+L2+L4": OWN + PATC + L2 + L4, "L2L4only": OWN + L2 + L4}


def fit(tr, te, cols):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                           objective="reg:absoluteerror", nthread=2, random_state=0)
    mdl.fit(tr[cols], tr.y); return mdl.predict(te[cols])


res = []
for v in range(2010, 2019):
    tr, te = D[D.t0 <= v - 3], D[D.t0 == v]
    r = {"v": v}
    for k, c in SETS.items():
        r[k] = np.abs(te.y - fit(tr, te, c)).mean()
    res.append(r)
R = pd.DataFrame(res)
out = {}
for k in SETS:
    out[k] = {"mae": round(R[k].mean(), 4), "skill_vsB1": round(float((1 - R[k] / R.B1).mean()), 4),
              "skill_vsM1": round(float((1 - R[k] / R.M1).mean()), 4), "win_vsM1": round(float((R[k] < R.M1).mean()), 2)}
print(json.dumps(out, ensure_ascii=False, indent=1))
R.to_csv(os.path.join(HERE, "g1_lag_result.csv"), index=False, encoding="utf-8-sig")

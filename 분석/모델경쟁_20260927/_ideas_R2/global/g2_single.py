# -*- coding: utf-8 -*-
"""G2 3차 — 특허 수준 피처 1개씩 B1에 추가(수출 3년 상대성장). 어떤 특허 포지션이 기여하는지 해석용. 개발 원점 2010..2018."""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
FLOW = sys.argv[1] if len(sys.argv) > 1 else "X"
D = pd.read_parquet(os.path.join(CACHE, f"g1_panel_{FLOW}.parquet"))
OWN = [c for c in ["lvl", "mom3", "mom1", "vol", "cn_sh", "cn_sh_chg", "imp_ratio", "imp_mom", "k_mom3"] if c in D]
LVL = [c for c in D.columns if c.startswith("pl_")] + ["p_domshare", "p_outshare", "p_krshare5"]
def fit(tr, te, cols):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                           objective="reg:absoluteerror", nthread=2, random_state=0)
    mdl.fit(tr[cols], tr.y); return mdl.predict(te[cols])
res = []
for v in range(2010, 2019):
    tr, te = D[D.t0 <= v - 3], D[D.t0 == v]
    r = {"v": v, "B1": np.abs(te.y - fit(tr, te, OWN)).mean()}
    for c in LVL:
        r[c] = np.abs(te.y - fit(tr, te, OWN + [c])).mean()
    res.append(r)
R = pd.DataFrame(res)
out = {c: (round(float((1 - R[c] / R.B1).mean()), 4), round(float((R[c] < R.B1).mean()), 2)) for c in LVL}
print(pd.DataFrame(out, index=["skill_vsB1", "win"]).T.sort_values("skill_vsB1", ascending=False).to_string())

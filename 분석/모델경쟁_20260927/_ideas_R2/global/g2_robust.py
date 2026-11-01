# -*- coding: utf-8 -*-
"""G2 강건성 — 특허 해외점유 상위 3개 KSIC(26212 OLED, 26422 휴대폰, 26211 LCD) 품목을 평가에서 제외해도 특허 기여가 남는가.
학습은 전체(원점 ≤ v−3), 평가만 제외. 또한 KSIC 군집 부트스트랩(1,000회)으로 A4−B1 차이의 95% CI."""
import sys, os
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
D = pd.read_parquet(os.path.join(CACHE, "g1_panel_X.parquet")); m = load_map(); D["k5"] = D.hs6.map(m.groupby("hs6").k5.first())
OWN = ["lvl", "mom3", "mom1", "vol", "imp_ratio", "imp_mom", "k_mom3"]
LVL = [c for c in D.columns if c.startswith("pl_")] + ["p_domshare", "p_outshare", "p_krshare5"]
def fit(tr, te, c):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                           objective="reg:absoluteerror", nthread=2, random_state=0); mdl.fit(tr[c], tr.y); return mdl.predict(te[c])
P = []
for v in range(2010, 2019):
    tr, te = D[D.t0 <= v-3], D[D.t0 == v].copy()
    te["eB1"] = np.abs(te.y - fit(tr, te, OWN)); te["eA4"] = np.abs(te.y - fit(tr, te, OWN + LVL)); P.append(te)
P = pd.concat(P)
for lab, E in [("전체", P), ("상위3 KSIC 제외", P[~P.k5.isin(["26212", "26422", "26211"])])]:
    g = E.groupby("t0")[["eB1", "eA4"]].mean()
    print(lab, "skill_vsB1", round(float((1 - g.eA4 / g.eB1).mean()), 4), "win", round(float((g.eA4 < g.eB1).mean()), 2))
rng = np.random.default_rng(0); ks = P.k5.unique(); agg = P.groupby("k5")[["eB1", "eA4"]].sum(); bs = []
for _ in range(1000):
    s = agg.loc[rng.choice(ks, len(ks))].sum(); bs.append(1 - s.eA4 / s.eB1)
s = agg.sum(); print("pooled skill", round(1 - s.eA4 / s.eB1, 4), "KSIC군집 95%CI", np.round(np.percentile(bs, [2.5, 97.5]), 4))

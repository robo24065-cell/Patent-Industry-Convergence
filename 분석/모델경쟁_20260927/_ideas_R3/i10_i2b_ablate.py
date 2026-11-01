# -*- coding: utf-8 -*-
"""I2b 해석용 묶음 분해(개발 원점만, 채택 판단에 쓰지 않음): M0 + 각 특허 묶음 하나씩."""
import os
import pandas as pd, numpy as np
from i00_common import *
D = pd.read_parquet(os.path.join(CACHE, "I2b_panel.parquet"))
M0 = ["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type"]
G = {"filing": ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y"],
     "maintenance": ["log_live", "lapse_1y", "lapse_2y", "pay_1y", "pay_trend", "live_ratio"],
     "transfer": ["trf_2y"]}
V = [2017, 2018, 2019, 2020, 2021]
base = {}
rows = []
for name, feats in [("M0", M0)] + [("M0+" + g, M0 + f) for g, f in G.items()] + [("M1_all", M0 + sum(G.values(), []))]:
    r = {"model": name}
    for v in V:
        tr = D[D.t0 <= v - 2]; te = D[D.t0 == v]
        a = auc(te.y, fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float)))
        r[v] = a
        if name == "M0": base[v] = a
    r["mean_skill_vs_M0"] = np.mean([skill(r[v], base[v]) for v in V]); r["win"] = np.mean([r[v] > base[v] for v in V])
    rows.append(r)
# NPS 미가입 기업(3인 미만 등)만 따로
sub = []
for v in V:
    tr = D[D.t0 <= v - 2]; te = D[(D.t0 == v)]
    p0 = fit_pred(tr[M0].astype(float), tr.y, te[M0].astype(float)); p1 = fit_pred(tr[M0 + sum(G.values(), [])].astype(float), tr.y, te[M0 + sum(G.values(), [])].astype(float))
    m = te.nps_present.values == 0
    sub.append(dict(v=v, n_nonnps=int(m.sum()), pos_nonnps=int(te.y.values[m].sum()), rate_nonnps=te.y.values[m].mean(), rate_nps=te.y.values[~m].mean(),
                    auc0_nonnps=auc(te.y.values[m], p0[m]), auc1_nonnps=auc(te.y.values[m], p1[m]), auc0_nps=auc(te.y.values[~m], p0[~m]), auc1_nps=auc(te.y.values[~m], p1[~m])))
R = pd.DataFrame(rows); S = pd.DataFrame(sub)
print(R.round(4).to_string(index=False)); print(S.round(4).to_string(index=False))
R.to_csv(os.path.join(OUT, "I2b_ablate.csv"), index=False, encoding="utf-8-sig"); S.to_csv(os.path.join(OUT, "I2b_subgroup.csv"), index=False, encoding="utf-8-sig")

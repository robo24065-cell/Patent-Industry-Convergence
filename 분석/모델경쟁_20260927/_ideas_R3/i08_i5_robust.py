# -*- coding: utf-8 -*-
"""I5 견고성(개발 원점만, 채택 판단에 쓰지 않음): 대형 출원인(원점별 KR 26429 출원 상위 5) 안/밖에서 M1 이득이 유지되는가, 상위 100건 적중."""
import os
import pandas as pd, numpy as np
from i00_common import *

D = pd.read_parquet(os.path.join(CACHE, "I5_panel.parquet"))
M0 = ["app_age", "reg_t0", "claims_t0", "app_rate", "app_logn", "mg_rate", "mg_logn"]
M1 = M0 + ["fam_US", "fam_EP", "fam_JP", "fam_n"]
rows = []
for v in range(2014, 2021):
    tr = D[D.t0 <= v - 2]; te = D[D.t0 == v].copy()
    te["p0"] = fit_pred(tr[M0].astype(float), tr.y, te[M0].astype(float))
    te["p1"] = fit_pred(tr[M1].astype(float), tr.y, te[M1].astype(float))
    top5 = te.app.value_counts().head(5).index
    r = dict(v=v, n=len(te), pos=int(te.y.sum()), share_pos_top5=round(te[te.app.isin(top5)].y.sum() / max(1, te.y.sum()), 3))
    for nm, sub in [("all", te), ("top5", te[te.app.isin(top5)]), ("rest", te[~te.app.isin(top5)])]:
        a0, a1 = auc(sub.y, sub.p0), auc(sub.y, sub.p1)
        r[f"{nm}_pos"] = int(sub.y.sum()); r[f"{nm}_auc0"] = a0; r[f"{nm}_auc1"] = a1; r[f"{nm}_skill"] = skill(a1, a0)
    r["top100_hit_M0"] = int(te.sort_values("p0", ascending=False).y.head(100).sum())
    r["top100_hit_M1"] = int(te.sort_values("p1", ascending=False).y.head(100).sum())
    r["fam_US_rate_pos"] = te.loc[te.y == 1, "fam_US"].mean(); r["fam_US_rate_neg"] = te.loc[te.y == 0, "fam_US"].mean()
    rows.append(r)
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
print("평균 skill all/top5/rest", R.all_skill.mean().round(4), R.top5_skill.mean().round(4), R.rest_skill.mean().round(4))
R.to_csv(os.path.join(OUT, "I5_robust.csv"), index=False, encoding="utf-8-sig")

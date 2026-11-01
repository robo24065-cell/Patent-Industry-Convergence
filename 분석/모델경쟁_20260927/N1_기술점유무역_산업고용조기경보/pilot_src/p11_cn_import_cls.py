# -*- coding: utf-8 -*-
"""아이디어 C 재정식화: (a) 이진 사건 '대중 수입점유율 3년 +10%p 이상 급증' AUC, (b) 원점 내 순위상관.
기준선 = 무역만 로지스틱/XGB, 모형 = + KSIC 특허."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
P = pd.read_parquet(f"{CACHE}/panel_cnimp.parquet")
P["ev"] = (P.y >= 10).astype(int)
print("사건율", round(P.ev.mean(), 3), "사건 수", P.ev.sum())
BF = ["s0", "ds_past", "lw0", "gw_past"]
PF = ["mom_CN", "mom_KR", "gap_CN_KR", "sh_CN", "dsh_CN", "mom_ALL", "lvl_CN", "frgn_mom"]
def fp(tr, te, F, kind):
    if kind == "lr":
        mu, sd = tr[F].mean(), tr[F].std()+1e-9
        return LogisticRegression(C=0.3, max_iter=2000).fit((tr[F]-mu)/sd, tr.ev).predict_proba((te[F]-mu)/sd)[:, 1]
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=2, random_state=0).fit(tr[F], tr.ev).predict_proba(te[F])[:, 1]
res = []
for v in range(2011, 2019):
    tr, te = P[P.t0 <= v-3], P[P.t0 == v]
    r = {"v": v, "n": len(te), "ev": te.ev.sum()}
    for k in ["lr", "xgb"]:
        pb, pm = fp(tr, te, BF, k), fp(tr, te, BF+PF, k)
        r[f"B_{k}"], r[f"M_{k}"] = roc_auc_score(te.ev, pb), roc_auc_score(te.ev, pm)
    res.append(r)
R = pd.DataFrame(res); print(R.round(3).to_string())
for k in ["lr", "xgb"]:
    sk = (R[f"M_{k}"]-R[f"B_{k}"])/(1-R[f"B_{k}"])
    print(k, "기준선 AUC", round(R[f"B_{k}"].mean(), 4), "모형 AUC", round(R[f"M_{k}"].mean(), 4), "skill", round(sk.mean(), 4), "승률", (sk > 0).mean())

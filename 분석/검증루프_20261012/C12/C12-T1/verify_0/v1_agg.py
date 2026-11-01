# -*- coding: utf-8 -*-
"""v1 agg: 단일 시드 예측을 3개씩 묶어(공식과 같은 평균 앙상블) skill·CI(공식 절차 B=1000, 시드 2022) 계산.
사용: python v1_agg.py <scen> <k_from> <k_to>   묶음 k = 시드 (3k,3k+1,3k+2)   -> verify_0\\v1_agg_<scen>_<k_from>_<k_to>.json"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V = os.path.dirname(os.path.abspath(__file__))
scen, k0, k1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def boot(y, pm, pb, g, n=1000, seed=2022):
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(np.asarray(g), return_inverse=True)
    order = np.argsort(inv, kind="stable"); cuts = np.cumsum(np.bincount(inv))[:-1]
    idx = np.split(order, cuts); G = len(idx); out = []
    for _ in range(n):
        pick = rng.integers(0, G, G); i = np.concatenate([idx[j] for j in pick])
        if y[i].min() == y[i].max():
            continue
        a1 = roc_auc_score(y[i], pm[i]); a0 = roc_auc_score(y[i], pb[i]); out.append((a1 - a0) / (1 - a0))
    return np.percentile(out, [2.5, 97.5]).tolist()


R = {}; t = time.time()
for k in range(k0, k1):
    ss = (3 * k, 3 * k + 1, 3 * k + 2)
    p = [pd.read_parquet(os.path.join(V, "v1_preds", scen, f"seed_{s}.parquet")) for s in ss]
    y = p[0].y.values; g = p[0].app.values
    pc = np.mean([x.p_champ.values for x in p], axis=0); p0 = np.mean([x.p_M0.values for x in p], axis=0)
    a1 = roc_auc_score(y, pc); a0 = roc_auc_score(y, p0)
    R[str(ss)] = dict(auc_champ=a1, auc_M0=a0, skill=(a1 - a0) / (1 - a0), ci=boot(y, pc, p0, g))
    print(ss, round(R[str(ss)]["skill"], 4), [round(v, 4) for v in R[str(ss)]["ci"]], round(time.time() - t), flush=True)
json.dump(R, open(os.path.join(V, f"v1_agg_{scen}_{k0}_{k1}.json"), "w", encoding="utf-8"), indent=1)

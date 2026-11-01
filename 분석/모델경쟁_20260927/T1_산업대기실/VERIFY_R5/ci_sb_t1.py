# T1 강화 기준선(SBB) 대비 skill의 법인 군집 부트스트랩 95% CI(R=200, 공식과 같은 법인 단위 복원추출). 산출 ci_sb_t1.json
import json, pathlib, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V = pathlib.Path(__file__).resolve().parent; R3 = V.parent/'R3'
S = pd.read_parquet(V/'sens_t1_scores.parquet')
o = pd.read_parquet(R3/'runs'/'FROZEN_final'/'o2019.parquet', columns=['prior_any']); keep = ~o.prior_any.values
y = pd.read_parquet(V/'labels_c2019_official.parquet').y_u.values[keep]
bs = pd.read_parquet(R3/'cache_final'/'o2019'/'baseline_scores.parquet').loc[keep].reset_index(drop=True)
cols = {'model': S.s.values, 'SBB': S.SBB.values, 'xgb_y_u': bs.xgb_y_u.values, 'xgb_y': bs.xgb_y.values}
fr_, inv = np.unique(S.firm.values, return_inverse=True); order = np.argsort(inv, kind='stable')
st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
rng = np.random.RandomState(4242); A = []; t = time.time()
for r in range(200):
    idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
    A.append([roc_auc_score(yy, v[idx]) for v in cols.values()])
A = np.array(A); q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
bmax = A[:, 2:].max(1)
out = dict(R=200, skill_vs_SBB=q((A[:, 0] - A[:, 1]) / (1 - A[:, 1])), skill_vs_official_strongest_xgb=q((A[:, 0] - bmax) / (1 - bmax)),
           auc_model=q(A[:, 0]), auc_SBB=q(A[:, 1]), secs=round(time.time() - t))
json.dump(out, open(V/'ci_sb_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(out)

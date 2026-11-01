# 법인 군집 부트스트랩 95% CI (검증 c=2009): v31(엄격 최종) / v22(출원기준 참고) / 기준선
import sys, pandas as pd, numpy as np, json, time
from lib import *
from sklearn.metrics import roc_auc_score, average_precision_score
t0 = time.time(); R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
a = pd.read_parquet(CACHE/'v31_scores.parquet'); b = pd.read_parquet(CACHE/'v22_scores.parquet'); z = pd.read_parquet(CACHE/'v0_scores.parquet')
# v0 기준선 점수는 xgb였음 → 고정 기준선(logit) 재계산 대신, 강화 기준선 수치와 함께 v0 xgb/logit은 note로. 여기서는 동일정보 엄격 기준선(sb)과 비교.
y = a.y.values; nc = a.newc.values; firm = a.firm.values
S = {'v31': a.s.values, 'v22': b.s.values, 'strict_base': a.sb.values}
fr, inv = np.unique(firm, return_inverse=True)
order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr))); en = np.searchsorted(inv[order], np.arange(len(fr)), side='right')
rng = np.random.RandomState(0); res = {k: {m: [] for m in ['auc_all', 'pr_all', 'auc_new', 'pr_new']} for k in S}
for r in range(R):
    pick = rng.randint(0, len(fr), len(fr))
    idx = np.concatenate([order[st[i]:en[i]] for i in pick])
    yy = y[idx]; nn = nc[idx]
    for k, s in S.items():
        ss = s[idx]
        res[k]['auc_all'].append(roc_auc_score(yy, ss)); res[k]['pr_all'].append(average_precision_score(yy, ss))
        res[k]['auc_new'].append(roc_auc_score(yy[nn], ss[nn])); res[k]['pr_new'].append(average_precision_score(yy[nn], ss[nn]))
out = {}
for k in S:
    out[k] = {m: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))] for m, v in res[k].items()}
for m in ['auc_all', 'pr_all', 'auc_new', 'pr_new']:
    dlt = np.array(res['v31'][m]) - np.array(res['strict_base'][m]); out[f'diff_v31_vs_strictbase_{m}'] = [float(np.percentile(dlt, 2.5)), float(np.median(dlt)), float(np.percentile(dlt, 97.5))]
json.dump(out, open(T1/'bootstrap_ci.json', 'w'), indent=1)
for k, v in out.items(): print(k, {m: [round(x, 4) for x in q] for m, q in v.items()} if isinstance(v, dict) else [round(x, 4) for x in v])
print('R', R, 'time', round(time.time() - t0))

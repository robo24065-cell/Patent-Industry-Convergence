# verify_2: S0/S4/S5 독립 재채점(동결 코드 import 없이). 읽기 전용.
import sys, json, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
D = pathlib.Path(sys.argv[1]); R3 = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3])
off = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
auc_off = off['model']['auc_all']; sk_off = off['skill']['auc_all']; base = (auc_off - sk_off)/(1 - sk_off)
L = pd.read_parquet(D/'cache/final_labels.parquet')
keep = ~L.prior_any.values; y = L.y_u.values[keep]; firm = L.firm.values[keep]
S = {}
for v in ['S0', 'S4', 'S5']:
    o = pd.read_parquet(D/'runs'/v/'o2019.parquet', columns=['firm', 'k', 'px', 'pl'])
    assert (o.firm.values == L.firm.values).all() and (o.k.values == L.k.values).all()
    o = o[keep]
    S[v] = 0.9*rankdata(o.px.values)/len(o) + 0.1*rankdata(o.pl.values)/len(o)
res = dict(base_auc_implied=base, n=int(len(y)), pos=int(y.sum()))
for v in S:
    a = roc_auc_score(y, S[v]); res[v] = dict(auc=a, skill=(a-base)/(1-base))
for v in ['S4', 'S5']:
    res[v]['d_skill_point'] = res[v]['skill'] - res['S0']['skill']
# 독립 법인 군집 부트스트랩(다른 난수, 기준선 AUC 고정 — 원 분석은 재표본별 기준선 사용)
fr_, inv = np.unique(firm, return_inverse=True)
order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fr_))); en = np.searchsorted(inv[order], np.arange(len(fr_)), side='right')
rng = np.random.RandomState(777); R = int(sys.argv[4]) if len(sys.argv) > 4 else 60
d = {v: [] for v in ['S4', 'S5']}
for r in range(R):
    idx = np.concatenate([order[st[i]:en[i]] for i in rng.randint(0, len(fr_), len(fr_))]); yy = y[idx]
    a0 = roc_auc_score(yy, S['S0'][idx])
    for v in d: d[v].append((roc_auc_score(yy, S[v][idx]) - a0)/(1 - base))
for v in d:
    a = np.array(d[v]); res[v]['d_skill_boot_fixedbase'] = [float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5))]; res[v]['R'] = R
json.dump(res, open(out, 'w', encoding='utf-8'), indent=1)
print(json.dumps(res, indent=1))

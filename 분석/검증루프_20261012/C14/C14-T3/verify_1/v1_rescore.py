# -*- coding: utf-8 -*-
"""verify_1 (C14-T3 반박): 최종·개발 Δskill 독립 재계산.
 - 최강 기준선: baseline_scores.parquet 각 열을 sklearn AUC로 직접 채점해 최대(공식 json 역산값 대신).
 - 모형 점수: 0.9*rank(px)/n + 0.1*rank(pl)/n (scipy rankdata, 동결 blend 미사용).
 - 개발: runs/<var>/o<c>.parquet vs FROZEN_dev, 기준선은 R3 cache/o<c>/baselines_r3.json strongest.auc_all.
 쓰기: verify_1/v1_rescore_out.json 만."""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
HERE = pathlib.Path(__file__).resolve().parent; C = HERE.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
R3 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'
L = pd.read_parquet(B/'분석/검증루프_20261012/C10/C10-T1/cache/final_labels.parquet')
keep = ~L.prior_any.values; y = L.y_u.values[keep].astype(int)
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet').loc[keep]
base = {k: roc_auc_score(y, bs[k].values) for k in bs.columns}
b = max(base.values())
def sc(o):
    n = len(o); return 0.9*rankdata(o.px.values)/n + 0.1*rankdata(o.pl.values)/n
def sk(a, bb): return (a-bb)/(1-bb)
OUT = dict(base_aucs=base, base_max=b, n=int(keep.sum()), pos=int(y.sum()))
def fin(p):
    o = pd.read_parquet(p, columns=['firm', 'k', 'px', 'pl'])
    assert (o.firm.values == L.firm.values).all() and (o.k.values == L.k.values).all()
    a = roc_auc_score(y, sc(o[keep])); return a, sk(a, b)
a0, s0 = fin(R3/'runs/FROZEN_final/o2019.parquet'); OUT['S0'] = dict(auc=a0, skill=s0)
V = ['P0', 'S1', 'S1v', 'S2', 'S3_1', 'S3_2', 'S3_3', 'S3_4', 'S3_5', 'S4', 'AN_A', 'AN_B', 'AN_TI']
fs = {}
for v in V:
    a, s = fin(C/'runs'/v/'o2019.parquet'); fs[v] = s - s0; OUT[v] = dict(auc=a, skill=s, d=s - s0)
OUT['S3_mean'] = float(np.mean([fs[f'S3_{i}'] for i in range(1, 6)]))
OUT['AN_BmA'] = fs['AN_B'] - fs['AN_A']
OUT['est'] = max(abs(OUT['S3_mean']), abs(fs['S4']), abs(OUT['AN_BmA']))
# dev
dev = {}
for c in range(2008, 2013):
    bj = json.load(open(R3/f'cache/o{c}/baselines_r3.json', encoding='utf-8'))
    bb = bj['strongest']['auc_all']
    def dv(p):
        o = pd.read_parquet(p, columns=['px', 'pl', 'y_u', 'prior_any']); o = o[~o.prior_any.values]
        return sk(roc_auc_score(o.y_u.values.astype(int), sc(o)), bb)
    f0 = dv(R3/f'runs/FROZEN_dev/o{c}.parquet')
    for v in V:
        p = C/'runs'/v/f'o{c}.parquet'
        if p.exists(): dev.setdefault(v, {})[c] = dv(p) - f0
OUT['dev'] = dev
dm = {v: float(np.mean(list(d.values()))) for v, d in dev.items()}
dm['S3_mean'] = float(np.mean([dm[f'S3_{i}'] for i in range(1, 6)]))
dm['AN_BmA'] = float(np.mean([dev['AN_B'][c] - dev['AN_A'][c] for c in dev['AN_A']]))
OUT['dev_mean'] = dm
fd = {'S1': fs['S1'], 'S2': fs['S2'], 'S3_mean': OUT['S3_mean'], 'S4': fs['S4'], 'AN_BmA': OUT['AN_BmA']}
OUT['final_minus_dev'] = {k: fd[k] - dm[k] for k in fd}
json.dump(OUT, open(HERE/'v1_rescore_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ['base_max', 'S0', 'S3_mean', 'AN_BmA', 'est', 'dev_mean', 'final_minus_dev']}, ensure_ascii=False, indent=1, default=float))
for v in V: print(v, round(OUT[v]['d'], 6))

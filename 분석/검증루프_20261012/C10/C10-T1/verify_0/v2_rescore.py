# -*- coding: utf-8 -*-
"""verify_0 v2 (2차 반박 검증): 작업자 스크립트·캐시 레이블을 쓰지 않는 독립 재채점.
레이블 = 공식 FROZEN_final/o2019.parquet 의 y_u, prior_any. AUC = Mann-Whitney(평균 순위) 직접 구현.
blend = 0.9*pct_rank(px) + 0.1*pct_rank(pl) (pandas rank pct, 평균 동점).
부트스트랩: 작업자와 다른 난수(RandomState(777)), 법인 군집 R=300, 기준선은 공식 최강 0.81078 고정(근사)."""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C10/C10-T1")
R3 = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/R3")
OUT = HERE/'verify_0'
BASE = 0.8107800399723293

def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def blend(o):
    return 0.9 * pd.Series(o.px.values).rank(pct=True).values + 0.1 * pd.Series(o.pl.values).rank(pct=True).values

res = {}
off = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet')
L = pd.read_parquet(HERE/'cache/final_labels.parquet')   # 작업자 레이블: 공식 json의 pos·AUC로 검증
assert (L.firm.values == off.firm.values).all() and (L.k.values == off.k.values).all()
keep = ~L.prior_any.values.astype(bool)
y = L.y_u.values[keep].astype(int); firm = off.firm.values[keep]
offj = json.load(open(R3/'final_result_r3.json', encoding='utf-8'))
print('pos', int(y.sum()), 'official pos', offj['n']['pos'], 'rows', int(keep.sum()), offj['n']['rows'])
S = {}
for v in ['OFFICIAL', 'S0', 'S1', 'S2', 'S3']:
    o = off if v == 'OFFICIAL' else pd.read_parquet(HERE/'runs'/v/'o2019.parquet')
    assert (o.firm.values == off.firm.values).all() and (o.k.values == off.k.values).all()
    ok = o[keep].reset_index(drop=True); s = blend(ok); S[v] = s
    a = auc(y, s); res[v] = dict(auc=a, skill=(a - BASE) / (1 - BASE), px_diff_frac=float((o.px.values != off.px.values).mean()))
for v in ['S1', 'S2', 'S3']:
    res[v]['d_skill'] = res[v]['skill'] - res['S0']['skill']
print(json.dumps(res, indent=1))

# 독립 부트스트랩
fu, inv = np.unique(firm, return_inverse=True)
order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(fu))); en = np.searchsorted(inv[order], np.arange(len(fu)), side='right')
rng = np.random.RandomState(777); R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
D = {v: [] for v in ['S1', 'S2', 'S3']}
for r in range(R):
    pick = rng.randint(0, len(fu), len(fu))
    idx = np.concatenate([order[st[i]:en[i]] for i in pick]); yy = y[idx]
    a0 = auc(yy, S['S0'][idx])
    for v in D: D[v].append((auc(yy, S[v][idx]) - a0) / (1 - BASE))
boot = {v: dict(lo=float(np.percentile(D[v], 2.5)), med=float(np.median(D[v])), hi=float(np.percentile(D[v], 97.5)), frac_lt0=float((np.array(D[v]) < 0).mean())) for v in D}
print(json.dumps(boot, indent=1))

# 개발 원점
dev = []
for c in (2008, 2009, 2010, 2011, 2012):
    f0 = pd.read_parquet(R3/'runs/FROZEN_dev'/f'o{c}.parquet'); k0 = ~f0.prior_any.values.astype(bool)
    y0 = f0.y_u.values[k0].astype(int); a0 = auc(y0, blend(f0[k0]))
    row = dict(origin=c, auc_frozen=a0)
    for v in ['S1', 'S2', 'S3']:
        o = pd.read_parquet(HERE/'runs'/v/f'o{c}.parquet')
        assert (o.firm.values == f0.firm.values).all() and (o.k.values == f0.k.values).all() and (o.y_u.values == f0.y_u.values).all()
        row[f'd_auc_{v}'] = auc(y0, blend(o[k0])) - a0
    dev.append(row)
dev = pd.DataFrame(dev); print(dev.to_string())
json.dump(dict(final=res, boot=boot, dev=dev.to_dict('records')), open(OUT/'v2_rescore_out.json', 'w', encoding='utf-8'), indent=1)

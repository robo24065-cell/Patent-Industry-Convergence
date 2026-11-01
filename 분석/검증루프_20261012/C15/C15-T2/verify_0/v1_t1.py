# -*- coding: utf-8 -*-
# python -I v1_t1.py <D> <m0|sbb> <b_lo> <b_hi>
import sys, json, time, numpy as np, pandas as pd
D, mode, blo, bhi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
T0 = time.time()
BS = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T1_산업대기실/R3/cache_final/o2019/baseline_scores.parquet'
def prep(x):
    u, inv = np.unique(x, return_inverse=True); return inv.astype(np.int32), len(u)
def wauc(pr, y, w):
    inv, U = pr; wp = np.bincount(inv, w * y, U); wn = np.bincount(inv, w * (1 - y), U)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
L = pd.read_parquet(f'{D}/cache/t1/final_labels.parquet'); keep = ~L.prior_any.values
S0 = pd.read_parquet(f'{D}/cache/t1/scores_b0.parquet', columns=['firm', 'k', 'y'])
assert (S0.firm.values == L.firm.values[keep]).all() and (S0.k.values == L.k.values[keep]).all()
assert (S0.y.values == L.y_u.values[keep]).all()
y = S0.y.values.astype(float); fr, finv = np.unique(S0.firm.values, return_inverse=True); NF = len(fr); del S0
one = np.ones(len(y))
cols = {}
if mode == 'm0':
    B = pd.read_parquet(BS, columns=['logit_stack_y', 'logit_single_y', 'logit_stack_y_u', 'logit_single_y_u']).loc[keep]
    for c in B.columns: cols[('off', c)] = prep(B[c].values)
    del B
for b in range(blo, bhi + 1):
    S = pd.read_parquet(f'{D}/cache/t1/scores_b{b}.parquet', columns=['firm', 's', 'xgb_y', 'xgb_y_u', 'sbb'])
    assert (fr[finv] == S.firm.values).all()
    cols[(b, 's')] = prep(S.s.values)
    if mode == 'm0':
        cols[(b, 'xgb_y')] = prep(S.xgb_y.values); cols[(b, 'xgb_y_u')] = prep(S.xgb_y_u.values)
    else:
        cols[(b, 'sbb')] = prep(S.sbb.values)
    del S
print('prep', round(time.time() - T0), flush=True)
point = {str(k): wauc(v, y, one) for k, v in cols.items()}
R = {str(k): [] for k in cols}
def rep(draw):
    w = np.bincount(draw, minlength=NF).astype(float)[finv]
    for k, v in cols.items(): R[str(k)].append(wauc(v, y, w))
if mode == 'm0':
    for ci in range(20):
        rng = np.random.RandomState(3000 + ci)
        for r in range(10): rep(rng.randint(0, NF, NF))
        print('chunk', ci, round(time.time() - T0), flush=True)
else:
    rng = np.random.RandomState(4242)
    for r in range(200):
        rep(rng.randint(0, NF, NF))
        if r % 20 == 19: print('rep', r, round(time.time() - T0), flush=True)
json.dump(dict(point=point, reps=R), open(f'{D}/verify_0/v1_t1_{mode}_{blo}_{bhi}.json', 'w'))
print('done', round(time.time() - T0), flush=True)

# -*- coding: utf-8 -*-
"""C18-T1 s6: T1 법인 군집 짝지은 부트스트랩. 인자: mode(off|c18) start n
 - off: 공식 재표본 규칙(RandomState(3000+ci), chunk 10) 재현 — start=ci 시작, n=chunk 수
 - c18: 사전등록 B=1000(시드 20261017+b), start=b 시작, n=개수
가중 AUC(재표본 중복=가중치)로 모형·6기준선 × 레이블(y0, S1, S2) 계산. 쓰기: cache/s6_boot_{mode}_{start}.parquet"""
import sys, pathlib, time
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
t0 = time.time()
mode, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
H = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C18/C18-T1"); C = H/'cache'
BCOLS = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
ev = pd.read_parquet(C/'s5_eval.parquet'); L = np.load(C/'s5_labels.npz')
Y = {'y0': L['y0'].astype(np.float64), 'S1': L['S1_R_self_G'].astype(np.float64), 'S2': L['S2_R_any_G'].astype(np.float64)}
S = {'model': ev.s.values}; S.update({c: ev['b_' + c].values for c in BCOLS})
prep = {}
for kk, sc in S.items():
    o = np.argsort(sc, kind='mergesort'); ss = sc[o]; st = np.r_[0, np.nonzero(np.diff(ss))[0] + 1]
    prep[kk] = (o, st, {l: y[o] for l, y in Y.items()})
def wauc(ws, ys, st):
    Gp = np.add.reduceat(ws * ys, st); Gn = np.add.reduceat(ws, st) - Gp
    below = np.cumsum(Gn) - Gn
    return float((Gp * (below + 0.5 * Gn)).sum() / (Gp.sum() * Gn.sum()))
def all_aucs(w):
    out = {}
    for kk, (o, st, ys) in prep.items():
        ws = w[o]
        for l, y_ in ys.items(): out[f'{kk}|{l}'] = wauc(ws, y_, st)
    return out
fr_, inv = np.unique(ev.firm.values, return_inverse=True); nf = len(fr_)
rows = []
if mode == 'off':
    for ci in range(start, start + n):
        rng = np.random.RandomState(3000 + ci)
        for r in range(10):
            w = np.bincount(rng.randint(0, nf, nf), minlength=nf)[inv].astype(np.float64)
            d = all_aucs(w); d['rep'] = ci * 10 + r; rows.append(d)
else:
    for b in range(start, start + n):
        rng = np.random.default_rng(20261017 + b)
        w = np.bincount(rng.integers(0, nf, nf), minlength=nf)[inv].astype(np.float64)
        d = all_aucs(w); d['rep'] = b; rows.append(d)
        if time.time() - t0 > 500: print('time guard stop at', b, flush=True); break
pd.DataFrame(rows).to_parquet(C/f's6_boot_{mode}_{start}.parquet', index=False)
print('done', mode, start, len(rows), round(time.time() - t0))

# 저장 모형 재예측(predict_saved)이 학습 직후 평가행 예측과 같은지 확인 (FUT, L=2025-07-01)
import os, sys, json, glob, numpy as np, pandas as pd
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OD = os.path.join(R2, 'runs_future', 'FUT')
spec = json.load(open(os.path.join(R2, 'final_spec_r2.json'), encoding='utf-8'))['spec']; W = spec['ensemble']['weights']
ev = pd.read_parquet(os.path.join(OD, 'ev.parquet'), columns=['pid'])
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
pf = {}
n = len(ev)
row = pd.read_parquet(os.path.join(OD, 'ev_haz.parquet'), columns=['row']).row.values
for c in list(W) + ['base']:
    if c in ('cls', 'haz'):
        z = np.mean([np.load(f) for f in sorted(glob.glob(os.path.join(OD, 'pred', f'{c}_*_b*.npy')))], 0)
        h = 1 / (1 + np.exp(-z))
        if c == 'haz':
            s_ = pd.Series(np.log1p(-np.clip(h, 1e-6, 1 - 1e-6))).groupby(row).sum()
            P = np.zeros(n); P[s_.index.values] = 1 - np.exp(s_.values); h = P
        pf[c] = h
    else:
        pf[c] = np.load(glob.glob(os.path.join(OD, 'pred', f'{c}_*.npy'))[0])
p = 1 / (1 + np.exp(-sum(W[c] * lg(pf[c]) for c in W) / sum(W.values())))
d = pd.read_parquet(os.path.join(OD, 'pred_20250701.parquet'))
assert (d.pid.values == ev.pid.values).all()
out = {c: float(np.max(np.abs(d['p_' + c].values - pf[c]))) for c in W}
out['ens'] = float(np.max(np.abs(d.p.values - p))); out['base'] = float(np.max(np.abs(d.p_base.values - pf['base'])))
out['corr_ens'] = float(np.corrcoef(d.p.values, p)[0, 1]); out['n'] = len(d)
print(out)
json.dump(out, open(os.path.join(R2, 'audit', 'future_consistency.json'), 'w'), indent=1)

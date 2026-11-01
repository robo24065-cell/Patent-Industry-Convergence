# verify_2 w1: P0(c-i) 독립 재계산. C12 저장 예측에서 sklearn 가중 AUC로 직접 계산(작업 AUCPrep 미사용).
import sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
C12C = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T4\cache'
OUT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T4\verify_2\w1_out.json'
ds = lambda a, b: (a - b) / (1 - b)
res = {}
for v in [2020, 2021, 2022, 2023, 2024]:
    base = pd.read_parquet(C12C + rf'\pred_{v}_base.parquet')
    pa = pd.read_parquet(C12C + rf'\pred_{v}_pubarms.parquet')
    rf = pd.read_parquet(C12C + rf'\pred_{v}_ref.parquet')
    for q in (pa, rf):
        assert (q.aid.values == base.aid.values).all() and (q.s.values == base.s.values).all() and (q.cc.values == base.cc.values).all()
    y, w = base.y.values, base.w.values
    A = {m: float(roc_auc_score(y, s, sample_weight=w)) for m, s in
         [('prov', base.prov), ('pub', base['pub']), ('free', pa.free), ('hsc', pa.hsc), ('all', rf['all']), ('pubctx', rf.pubctx)]}
    A['npos'] = int(y.sum()); A['n'] = len(y)
    A['E1'] = ds(A['all'], A['pubctx']); A['E2'] = ds(A['all'], A['prov'])
    A['dAUC'] = A['prov'] - max(A['free'], A['hsc'])
    res[v] = A
    print(v, {k: round(x, 5) if isinstance(x, float) else x for k, x in A.items()}, flush=True)
res['E1_mean'] = float(np.mean([res[v]['E1'] for v in range(2020, 2025)]))
res['E2_mean'] = float(np.mean([res[v]['E2'] for v in range(2020, 2025)]))
res['n_dAUC_le0'] = int(sum(res[v]['dAUC'] <= 0 for v in range(2020, 2025)))
print(res['E1_mean'], res['E2_mean'], res['n_dAUC_le0'])
json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1)

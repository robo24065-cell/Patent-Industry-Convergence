# -*- coding: utf-8 -*-
"""R8 s04b: 직전 실행(s04)이 conc_s3 완료 전에 끝나 빠진 쌍만 추가 계산한다.
 prov_s3 vs conc_s3 (XGB 시드 3개 평균, 강건성), prov_s3 vs none, conc_s3 vs none. CI 방식은 s04와 동일(원점 내 mg 복원추출 1,000회)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(HERE, 'results'); B = 1000
rng = np.random.default_rng(20261005); ORIG = list(range(2010, 2019))
def auc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def paired(A, Bd):
    aA, aB, bs = [], [], np.zeros((len(ORIG), B))
    for i, v in enumerate(ORIG):
        a = A[A.t0 == v].sort_values('cell'); b = Bd[Bd.t0 == v].sort_values('cell')
        assert (a.cell.values == b.cell.values).all() and (a.yb.values == b.yb.values).all()
        y = a.yb.values; pa = a.p.values; pb = b.p.values; aA.append(auc(y, pa)); aB.append(auc(y, pb))
        I = rng.integers(0, len(y), (B, len(y)))
        for j in range(B):
            ii = I[j]; yy = y[ii]
            bs[i, j] = np.nan if yy.sum() in (0, len(yy)) else auc(yy, pa[ii]) - auc(yy, pb[ii])
    d = np.array(aA) - np.array(aB); bm = np.nanmean(bs, 0)
    return dict(aucA=float(np.mean(aA)), aucB=float(np.mean(aB)), dAUC=float(d.mean()),
                ci=[float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))], wins=int((d > 0).sum()), n_orig=9,
                by_origin=[round(float(x), 4) for x in d])
L = lambda t: pd.read_parquet(os.path.join(RD, f'mg_ctx-{t}_preds.parquet'))
M = {t: L(t) for t in ['prov_s3', 'conc_s3', 'none']}
R = {}
for A, Bn in [('prov_s3', 'conc_s3'), ('prov_s3', 'none'), ('conc_s3', 'none')]:
    r = paired(M[A], M[Bn]); R[f'{A}_vs_{Bn}'] = r
    print(f"mg {A} vs {Bn}: {r['aucA']:.4f} vs {r['aucB']:.4f} d{r['dAUC']:+.4f} CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] wins {r['wins']}/9")
json.dump(R, open(os.path.join(HERE, 'summary_r8_s3pair.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')

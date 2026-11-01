# C17-T4 verify_1 w1: 저장된 개발 예측(pred_*.parquet)에서 AUC를 자체 순위 공식으로 재계산(sklearn 미사용),
# 원점별 '최선 공개'(원점별 최댓값 vs 평균 최선=OFF) 차이, PROV-BASE / 공개-BASE 의 KSIC 군집 부트스트랩 SE(B=300).
# 읽기만: cache\pred_*.parquet. 쓰기: verify_1\w1_out.json
import json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C17\C17-T4'
ARMS = ['base', 'prov', 'free', 'hsc', 'off']


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_w(y, p, w):
    # 가중 AUC(군집 부트스트랩용, 동점은 0.5): 정렬 후 누적
    o = np.argsort(p, kind='mergesort'); y = y[o]; p = p[o]; w = w[o]
    ps, idx = np.unique(p, return_index=True)
    w1 = np.where(y == 1, w, 0.0); w0 = np.where(y == 0, w, 0.0)
    g1 = np.add.reduceat(w1, idx); g0 = np.add.reduceat(w0, idx)
    c0 = np.cumsum(g0) - g0
    num = (g1 * (c0 + 0.5 * g0)).sum()
    return num / (g1.sum() * g0.sum())


res = {}
rng = np.random.default_rng(20261017)
for lr in ['xgb', 'lr']:
    for Y in (2021, 2022, 2023):
        P = pd.read_parquet(W + rf'\cache\pred_{Y}_{lr}.parquet')
        y = P.y.values.astype(int)
        a = {k: float(auc(y, P[k].values)) for k in ARMS}
        best_pub_origin = max(['free', 'hsc', 'off'], key=lambda k: a[k])
        # KSIC 군집 부트스트랩
        ks = P.k_prov.astype(str).values
        cats, inv = np.unique(ks, return_inverse=True)
        diffs = {'prov-base': [], 'prov-off': [], 'off-base': [], 'hsc-base': [], 'free-base': [], 'prov-bestorigin': []}
        for b in range(300):
            cnt = np.bincount(rng.integers(0, len(cats), len(cats)), minlength=len(cats)).astype(float)
            w = cnt[inv]
            m = w > 0
            ab = {k: auc_w(y[m], P[k].values[m], w[m]) for k in ARMS}
            diffs['prov-base'].append(ab['prov'] - ab['base']); diffs['prov-off'].append(ab['prov'] - ab['off'])
            diffs['off-base'].append(ab['off'] - ab['base']); diffs['hsc-base'].append(ab['hsc'] - ab['base'])
            diffs['free-base'].append(ab['free'] - ab['base']); diffs['prov-bestorigin'].append(ab['prov'] - ab[best_pub_origin])
        se = {k: float(np.std(v, ddof=1)) for k, v in diffs.items()}
        pt = {'prov-base': a['prov'] - a['base'], 'prov-off': a['prov'] - a['off'], 'off-base': a['off'] - a['base'],
              'hsc-base': a['hsc'] - a['base'], 'free-base': a['free'] - a['base'], 'prov-bestorigin': a['prov'] - a[best_pub_origin]}
        # 단계별 PROV-BASE
        st = {}
        for s in (2, 3):
            mm = P.s.values == s
            st[f's{s}'] = {'n': int(mm.sum()), 'prov-base': float(auc(y[mm], P.prov.values[mm]) - auc(y[mm], P.base.values[mm])),
                           'prov-off': float(auc(y[mm], P.prov.values[mm]) - auc(y[mm], P.off.values[mm]))}
        res[f'{lr}_{Y}'] = {'n': len(P), 'pos': int(y.sum()), 'auc': a, 'best_pub_origin': best_pub_origin,
                            'diff': pt, 'se_ksic': se, 'z': {k: pt[k] / se[k] for k in pt}, 'by_stage': st}
        print(lr, Y, {k: round(v, 5) for k, v in a.items()}, best_pub_origin,
              {k: (round(pt[k], 5), round(se[k], 5)) for k in pt}, flush=True)
    m = {k: float(np.mean([res[f'{lr}_{Y}']['auc'][k] for Y in (2021, 2022, 2023)])) for k in ARMS}
    res[f'{lr}_mean'] = m
    print(lr, 'mean', {k: round(v, 5) for k, v in m.items()}, flush=True)
json.dump(res, open(W + r'\verify_1\w1_out.json', 'w'), indent=1)

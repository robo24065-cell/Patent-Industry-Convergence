# 외부데이터 신호 점검: KOSIS 광업제조업조사(KR 세세분류 5자리) 출하·부가가치·종사자 성장 → 특허 출원 성장(국가별)
# 목표연도 ≤2018 (검증 구간 자료만)
import numpy as np, pandas as pd, os
from lab2 import D, FR
K = pd.read_csv(os.path.join(FR.T, '..', '..', '..', '공모전데이터', '작업데이터', 'kosis_광업제조업_50개산업.csv'), encoding='utf-8-sig', dtype={'세세분류코드': str})
K = K[K.기준 == '10차']
ks = list(D['idx'].get_level_values(0).unique())
def ser(item):
    p = K[K.항목코드 == item].pivot_table(index='세세분류코드', columns='연도', values='값').reindex(ks)
    p[2010] = np.sqrt(p[2009] * p[2011]); p = p.reindex(columns=range(2008, 2019))
    return np.log(p.clip(lower=1))
L = np.log1p(D['n'])
for item, nm in [('T04', '출하액'), ('T06', '부가가치'), ('T02', '종사자'), ('T07', '유형자산')]:
    S = ser(item)
    for lagx in [0, 1]:
        out = {}
        for j, c in enumerate(FR.CCS):
            rows = np.where(D['CCI'] == j)[0]; kidx = [ks.index(D['idx'][i][0]) for i in rows]
            yy, xx, zz = [], [], []
            for t in range(2011, 2018):  # 특허 성장 t→t+1, 생산 성장 (t-lagx-1 → t-lagx)
                a = L[rows, t + 1 - 2000] - L[rows, t - 2000]
                own = L[rows, t - 2000] - L[rows, t - 1 - 2000]
                x = (S[t - lagx] - S[t - lagx - 1]).values[kidx]
                ok = np.isfinite(x)
                yy.append(a[ok] - a[ok].mean()); xx.append(x[ok] - x[ok].mean()); zz.append(own[ok] - own[ok].mean())
            yy, xx, zz = map(np.concatenate, (yy, xx, zz))
            X = np.column_stack([np.ones_like(xx), xx, zz]); b = np.linalg.lstsq(X, yy, rcond=None)[0]
            r = yy - X @ b; se = np.sqrt(np.diag(r @ r / (len(yy) - 3) * np.linalg.inv(X.T @ X)))
            out[c] = f'{b[1]:+.3f}(t={b[1]/se[1]:+.1f})'
        print(nm, 'lag', lagx, out)

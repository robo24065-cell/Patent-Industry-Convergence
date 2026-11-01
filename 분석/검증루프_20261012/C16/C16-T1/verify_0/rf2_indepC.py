# -*- coding: utf-8 -*-
"""rf2: 독립 C(1,798) 부분집합으로 S1-full/S1-fixed·L-a 점추정 재계산(rf1 모집단 함수 재사용 없이 간단 재구성)."""
import pickle, json, pathlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; FC = T1/'R3/cache_final/o2019'; RD = T1/'R3/runs/FROZEN_final'; W = B/'분석/검증루프_20261012/C16/C16-T1'
P = pickle.load(open(W/'verify_0/v0_pop.pkl', 'rb'))
o = pd.read_parquet(RD/'o2019.parquet', columns=['firm', 'k', 'prior_any', 'px', 'pl'])
o['y'] = pd.read_parquet(B/'분석/검증루프_20261012/C15/C15-T2/cache/t1/final_labels.parquet', columns=['y_u']).y_u.values
bo = pd.read_parquet(FC/'baseline_scores.parquet'); BK = list(bo.columns)
for c in BK: o[c] = bo[c].values
S = pd.read_parquet(W/'cache/build/s1F/scores.parquet', columns=['firm', 'k', 'px', 'pl'] + BK)
Si = pd.Series(np.arange(len(S)), index=pd.MultiIndex.from_arrays([S.firm.values, S.k.values]))
pv = pd.MultiIndex.from_arrays([o.firm.values, o.k.values]).isin(pd.MultiIndex.from_tuples(list(P['prior_vis'])))
fin = o.firm.isin(P['Fvis']).values
m = (~o.prior_any.values & fin) | (o.prior_any.values & fin & ~pv)
D = o[m].copy(); D.loc[D.prior_any.values, 'y'] = 0
po = Si.reindex(pd.MultiIndex.from_arrays([D.firm.values, D.k.values])).values.astype(np.int64)
ci = Si.reindex(pd.MultiIndex.from_tuples(P['C'])).values.astype(np.int64)
Cd = S.iloc[ci].copy(); Cd['y'] = 0
rp = lambda x: pd.Series(x).rank(pct=True).values
def sk(DD):
    y = DD.y.values; s = 0.9 * rp(DD.px.values) + 0.1 * rp(DD.pl.values); M0 = max(roc_auc_score(y, DD[c].values) for c in BK); a = roc_auc_score(y, s)
    return (a - M0) / (1 - M0)
cols = ['px', 'pl', 'y'] + BK
fixed = pd.concat([D[cols], Cd[cols]]); Df = D.copy()
for c in ['px', 'pl'] + BK: Df[c] = S[c].values[po]
full = pd.concat([Df[cols], Cd[cols]])
OFF = 0.47267044268107444
r = dict(n=len(full), C=len(Cd), fixed=sk(fixed) - OFF, full=sk(full) - OFF)
print(r); json.dump(r, open(W/'verify_0/rf2_out.json', 'w'), indent=1)

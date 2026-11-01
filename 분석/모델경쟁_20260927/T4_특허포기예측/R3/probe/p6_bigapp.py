# -*- coding: utf-8 -*-
"""진단(모형 수정 없음): 개발 원점에서 상위 10개 출원인 집단 안/밖 AUC — E00 대 X21(출원인 가중 상한)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r3lib, r3run, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = r3lib.R
out = []
for o, _, _ in R.ORIGINS:
    ev = R.ev_meta(o); y = ev.y.values
    a = ev.acd.astype(str).values
    top = pd.Series(a[a != 'NA']).value_counts().index[:10]
    m = np.isin(a, top)
    pb = R.fit_predict_sk(o, r3run.load_spec('E00')['spec'], 'base')
    r = dict(origin=o, top10_share=round(float(m.mean()), 3))
    for e in ['E00', 'X21']:
        p, _ = R.ensemble(o, r3run.load_spec(e)['spec'])
        r[f'{e}_top10'] = round(roc_auc_score(y[m], p[m]), 4); r[f'{e}_rest'] = round(roc_auc_score(y[~m], p[~m]), 4)
    r['base_top10'] = round(roc_auc_score(y[m], pb[m]), 4); r['base_rest'] = round(roc_auc_score(y[~m], pb[~m]), 4)
    out.append(r)
df = pd.DataFrame(out); print(df.to_string())
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p6_bigapp.csv'), index=False, encoding='utf-8-sig')

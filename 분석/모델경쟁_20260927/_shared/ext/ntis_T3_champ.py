# -*- coding: utf-8 -*-
"""T3 챔피언 대비 증분 점검: R2 챔피언(E57) 개발 원점 예측(R2/preds/dev_preds.parquet, 읽기만)에 NTIS 셀 피처를 얹으면 나아지는가.
- (a) 원점별 in-sample: yb ~ logit(p_champ) + rank(x)  → 계수·ΔAUC
- (b) 표본 밖: v=2013..2016, 계수 β 를 t0 ≤ v-3 원점에서 추정해 v 에 적용(z = a + b·logit(p) + β·x) → ΔAUC, skill 차이(기준선 p_base 대비)
입력: ntis_T_work/ntis_T3_cellpanel.parquet (ntis_T3_test.py 산출). 🔒 t0 ≤2016 만.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from ntis_T_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P = pd.read_parquet(os.path.join(T3, 'R2', 'preds', 'dev_preds.parquet'))
assert P.t0.max() <= 2016
C = pd.read_parquet(out('ntis_T3_cellpanel.parquet'))
D = P.merge(C.drop(columns=['ksic', 'mg', 'y']), on=['cell', 't0'], how='inner', suffixes=('', '_n'))
print('결합', len(D), '/', len(P), 'yb 일치율', round(float((D.yb == D.yb_n).mean()), 4), flush=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
D['zc'] = lg(D.p)
FEATS = ['gsh', 'gsh_rel', 'gsh_pub', 'gsh_firm', 'gsh_chg', 'f_grow', 'f_int', 'k_a_grow', 'k_b2_grow', 'k_b1_grow']
for f in FEATS:
    D[f + '_r'] = D.groupby('t0')[f].transform(lambda s: rank_norm(s) if s.notna().sum() > 0 else s)

rows = []
for f in FEATS:
    for t0, g in D.groupby('t0'):
        g = g[g[f].notna()]
        if len(g) < 200 or g[f].nunique() < 5:
            continue
        m0 = LogisticRegression(C=1e6, max_iter=1000).fit(g[['zc']], g.yb)
        m1 = LogisticRegression(C=1e6, max_iter=1000).fit(g[['zc', f + '_r']], g.yb)
        a0 = roc_auc_score(g.yb, m0.predict_proba(g[['zc']])[:, 1]); a1 = roc_auc_score(g.yb, m1.predict_proba(g[['zc', f + '_r']])[:, 1])
        rows.append(dict(feat=f, t0=t0, n=len(g), coef=round(float(m1.coef_[0][1]), 4), d_auc_in=round(a1 - a0, 5)))
IN = pd.DataFrame(rows)

oos = []
for f in FEATS:
    for v in range(2013, 2017):
        tr = D[(D.t0 <= v - 3) & D[f].notna()]; te = D[(D.t0 == v)].copy()
        if len(tr) < 500 or te[f].notna().sum() < 200:
            continue
        m1 = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['zc', f + '_r']], tr.yb)
        te[f + '_r'] = te[f + '_r'].fillna(0.0)
        z1 = m1.decision_function(te[['zc', f + '_r']])
        a0 = roc_auc_score(te.yb, te.p); a1 = roc_auc_score(te.yb, z1); ab = roc_auc_score(te.yb, te.p_base)
        oos.append(dict(feat=f, v=v, n=len(te), auc_champ=round(a0, 4), auc_plus=round(a1, 4), d_auc=round(a1 - a0, 4),
                        d_skill=round((a1 - ab) / (1 - ab) - (a0 - ab) / (1 - ab), 4), beta=round(float(m1.coef_[0][1]), 3)))
O = pd.DataFrame(oos)
IN.to_csv(out('ntis_T3_champ_in.csv'), index=False, encoding='utf-8-sig'); O.to_csv(out('ntis_T3_champ_oos.csv'), index=False, encoding='utf-8-sig')
S = IN.groupby('feat').agg(n_orig=('t0', 'size'), coef=('coef', 'mean'), pos=('coef', lambda v: int((v > 0).sum())), d_auc_in=('d_auc_in', 'mean')).reset_index()
S = S.merge(O.groupby('feat').agg(oos_d_auc=('d_auc', 'mean'), oos_d_skill=('d_skill', 'mean'),
                                  oos_win=('d_auc', lambda v: f'{int((v > 0).sum())}/{len(v)}')).reset_index(), on='feat', how='left')
S.to_csv(out('ntis_T3_champ_summary.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(S.to_string()); print(O.to_string())

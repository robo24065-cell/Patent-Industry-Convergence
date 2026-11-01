# 감사 2: 스냅샷 패밀리 vs 시점보정 패밀리 — 분포·단변량 AUC·경과연수별 차이 (검증 구간만)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
cols = ['y', 'L', 'age', 'fam_n', 'fd_fam_n', 'fam_cc', 'fd_fam_cc', 'atype']
tr = pd.read_parquet('cache/s2_tr.parquet', columns=cols); ev = pd.read_parquet('cache/s2_ev.parquet', columns=cols)
for nm, d in [('train(L2005-09)', tr), ('eval(L2014-15)', ev)]:
    d['ageb'] = pd.cut(d.age, [0, 2, 4, 6, 8, 20])
    print(nm, 'n', len(d))
    print(d.groupby('ageb', observed=True)[['fam_n', 'fd_fam_n', 'fam_cc', 'fd_fam_cc']].mean().round(2).to_string())
    print(' share fam_n>0: snapshot', round((d.fam_n > 0).mean(), 3), 'dated', round((d.fd_fam_n > 0).mean(), 3))
    for c in ['fam_n', 'fd_fam_n', 'fam_cc', 'fd_fam_cc']:
        print('  univariate AUC(-x)', c, round(roc_auc_score(d.y, -d[c]), 4))
    # 레이블별 평균
    print(d.groupby('y')[['fam_n', 'fd_fam_n']].mean().round(2).to_string())

# 진단5: 신규진입자 법인선택/산업선택 분해(v31 vs v36), 가시이력 구간별 — 검증 c=2009만
import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
C = '../T1_산업대기실/cache/'
a = pd.read_parquet('cache/v31_scores.parquet'); b = pd.read_parquet('cache/v36_scores.parquet')
d = b.assign(s31=a.s.values)
def decomp(n, col):
    f = n.groupby('firm').agg(y=('y', 'max'), s=(col, 'max'))
    w = [roc_auc_score(g.y, g[col]) for _, g in n[n.firm.isin(f[f.y == 1].index)].groupby('firm') if 0 < g.y.sum() < len(g)]
    return round(roc_auc_score(f.y, f.s), 4), round(np.mean(w), 4)
n = d[d.newc]
print('신규진입자 [법인선택 AUC, 법인내 산업선택 AUC]  v31:', decomp(n, 's31'), ' v36:', decomp(n, 's'))
print('전체 [법인선택, 산업선택] v31:', decomp(d, 's31'), ' v36:', decomp(d, 's'))
d['vis'] = pd.cut(np.expm1(d.lsize).round(), [-1, 0, 2, 5, 20, 1e9], labels=['0', '1-2', '3-5', '6-20', '21+'])
print(d.groupby('vis', observed=True).apply(lambda g: pd.Series(dict(n=len(g), pos=int(g.y.sum()), v31=round(roc_auc_score(g.y, g.s31), 4), v36=round(roc_auc_score(g.y, g.s), 4)))).to_string())

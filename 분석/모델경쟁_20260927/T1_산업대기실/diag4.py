# 진단4: 엄격 최선(v31) vs 출원기준(v22) — 신규진입자 분해, 가시 이력 규모별, 산업별
import os; os.environ['T1_STRICT'] = '1'
from data8 import *
from sklearn.metrics import roc_auc_score, average_precision_score
a = pd.read_parquet(PC/'../v31_scores.parquet') if False else pd.read_parquet(CACHE/'v31_scores.parquet')
b = pd.read_parquet(CACHE/'v22_scores.parquet')
tr, te = load19((2004,))
d = te[['firm', 'k', 'y', 'newc', 'lsize', 'lout', 'lin']].assign(s=a.s.values, s22=b.s.values)
def decomp(n, col):
    f = n.groupby('firm').agg(y=('y', 'max'), s=(col, 'max'))
    w = [roc_auc_score(g.y, g[col]) for _, g in n[n.firm.isin(f[f.y == 1].index)].groupby('firm') if 0 < g.y.sum() < len(g)]
    h1 = np.mean([g.sort_values(col, ascending=False).y.iloc[0] for _, g in n[n.firm.isin(f[f.y == 1].index)].groupby('firm')])
    return round(roc_auc_score(f.y, f.s), 4), round(np.mean(w), 4), round(h1, 3)
n = d[d.newc]
print('신규진입자 [법인선택 AUC, 법인내 산업선택 AUC, top1 적중]  엄격 v31:', decomp(n, 's'), ' 출원기준 v22:', decomp(n, 's22'))
d['vis'] = pd.cut(np.expm1(d.lsize), [-1, 0, 5, 20, 1e9], labels=['0', '1-5', '6-20', '21+'])
def seg(g):
    r = dict(n=len(g), pos=int(g.y.sum()))
    if 5 <= g.y.sum() < len(g): r.update(auc_strict=round(roc_auc_score(g.y, g.s), 4), auc_filing=round(roc_auc_score(g.y, g.s22), 4))
    return pd.Series(r)
print('엄격 모드 가시 이력(건) 구간별'); print(d.groupby(['newc', 'vis'], observed=True).apply(seg).to_string())

# 진단2: 신규진입자 오차 분해 — 기업 선택(누가 진입하나) vs 산업 선택(어디로) / 세그먼트
from data8 import *
from sklearn.metrics import roc_auc_score, average_precision_score
tr, te = load14()
s = pd.read_parquet(CACHE/'v14_scores.parquet').sx.values; b = pd.read_parquet(CACHE/'v0_scores.parquet').s_base.values
d = te.assign(s=s, b=b)
n = d[d.newc]
f = n.groupby('firm').agg(y=('y', 'max'), s=('s', 'max'), b=('b', 'max'), sm=('s', 'sum'), nout=('lout', 'first'), tp=('firm', 'first'))
print('신규진입 법인', len(f), '5년내 진입 법인', int(f.y.sum()), f'({f.y.mean():.3f})')
print('기업 선택 AUC(법인 max 점수→어느 산업이든 진입): 모형', round(roc_auc_score(f.y, f.s), 4), '기준선', round(roc_auc_score(f.y, f.b), 4), '| 합점수', round(roc_auc_score(f.y, f.sm), 4))
w = []; wb = []
for fm, g in n[n.firm.isin(f[f.y == 1].index)].groupby('firm'):
    if 0 < g.y.sum() < len(g): w.append(roc_auc_score(g.y, g.s)); wb.append(roc_auc_score(g.y, g.b))
print('산업 선택 AUC(진입 법인 내부 50산업 순위, 법인 평균): 모형', round(np.mean(w), 4), '기준선', round(np.mean(wb), 4), 'n', len(w))
# 산업 선택 top-1/top-3 적중
hit1 = []; hit3 = []
for fm, g in n[n.firm.isin(f[f.y == 1].index)].groupby('firm'):
    o = g.sort_values('s', ascending=False); hit1.append(o.y.iloc[0] == 1); hit3.append(o.y.iloc[:3].max() == 1)
print('진입 법인 중 최상위 1개 산업 적중', round(np.mean(hit1), 3), '상위3', round(np.mean(hit3), 3))
tp = load()['fm'].set_index('applicant_cd').tp
n = n.assign(tp=n.firm.map(tp).fillna('?'), osz=pd.cut(np.expm1(n.lout), [0, 5, 20, 100, 1e9], labels=['3-5', '6-20', '21-100', '100+']))
def seg(g):
    r = dict(n=len(g), firms=g.firm.nunique(), pos=int(g.y.sum()))
    if 5 <= g.y.sum() < len(g): r.update(auc_m=round(roc_auc_score(g.y, g.s), 4), auc_b=round(roc_auc_score(g.y, g.b), 4), pr_m=round(average_precision_score(g.y, g.s), 4))
    return pd.Series(r)
print(n.groupby('tp').apply(seg).to_string()); print(n.groupby('osz', observed=True).apply(seg).to_string())
print(n.groupby('has_dart').apply(seg).to_string())

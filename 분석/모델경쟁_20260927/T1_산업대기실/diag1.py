# 진단1: 스태킹 기준선(강화) + v5 세그먼트별 오차(산업·규모·국가·DART 여부·신규/기존)
import pandas as pd, numpy as np
from exp import *
from sklearn.metrics import roc_auc_score, average_precision_score
tr, te, cte = get(train_cuts=(2002, 2003, 2004))
tr1, _, _ = get()
# 기준선 강화 시도: 스태킹 원점 학습 로짓/XGB
res = {}
res['logit_stack'] = metrics(te, logit(tr, te, BASE))
res['xgb_stack'] = metrics(te, predict(fit_xgb(tr, BASE), te, BASE))
res['logit_2004'] = metrics(te, logit(tr1, te, BASE))
for k, v in res.items(): print('baseline', k, fmt(v))
v5 = pd.read_parquet(CACHE/'v5_scores.parquet'); v0 = pd.read_parquet(CACHE/'v0_scores.parquet')
bl = logit(tr1, te, BASE)
d = te.copy(); d['s'] = v5.s.values; d['b'] = bl
fm = load()['fm'].set_index('applicant_cd').tp
d['tp'] = d.firm.map(fm).fillna('?')
d['size'] = pd.qcut(d.lsize, 3, labels=['S', 'M', 'L'])
def seg(g):
    if g.y.sum() < 5 or g.y.sum() == len(g): return pd.Series(dict(n=len(g), pos=g.y.sum()))
    return pd.Series(dict(n=len(g), pos=int(g.y.sum()), rate=g.y.mean(), auc_m=roc_auc_score(g.y, g.s), auc_b=roc_auc_score(g.y, g.b),
                          pr_m=average_precision_score(g.y, g.s), pr_b=average_precision_score(g.y, g.b)))
pd.set_option('display.width', 200)
for col in ['newc', 'tp', 'size', 'has_dart']:
    print(d.groupby(col, observed=True).apply(seg).round(4).to_string())
ks = d.groupby('k').apply(seg)
K = load()['K']; ks.index = [K[i] for i in ks.index]
print('KSIC별 AUC 모형 분포', ks.auc_m.describe().round(3).to_dict())
print('모형<기준선 KSIC 수', (ks.auc_m < ks.auc_b).sum())
print(ks.sort_values('auc_m').head(8).round(3).to_string())
ks.to_csv(T1/'diag_v5_by_ksic.csv', encoding='utf-8-sig')
json.dump(res, open(T1/'baseline_stack_diag.json', 'w'), indent=1)

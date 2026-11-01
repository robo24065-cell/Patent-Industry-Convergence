# -*- coding: utf-8 -*-
"""최종시험 사후 진단(모형은 고치지 않는다): 10분위 적중표, 보정, 대형 출원인 영향, 산업 순위 비교.
결과: R2/audit/diag_final.json"""
import os, sys, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OD = os.path.join(R2, 'runs_final', 'F22')
ev = pd.read_parquet(os.path.join(OD, 'ev.parquet'), columns=['pid', 'ksic', 'acd', 'atype', 'y', 'age'])
spec = json.load(open(os.path.join(R2, 'final_spec_r2.json'), encoding='utf-8'))['spec']
W = spec['ensemble']['weights']
pf = {}
for comp in list(W) + ['base']:
    fs = [f for f in glob.glob(os.path.join(OD, 'pred', f'{comp}_*.npy')) if '_b' not in os.path.basename(f)[len(comp) + 1:]]
    pf[comp] = np.load(fs[0])
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
p = 1 / (1 + np.exp(-sum(W[c] * lg(pf[c]) for c in W) / sum(W.values())))
ev['p'] = p; ev['pb'] = pf['base']
out = {}
# 10분위
ev['dec'] = pd.qcut(ev.p.rank(method='first', ascending=False), 10, labels=range(1, 11))
ev['decb'] = pd.qcut(ev.pb.rank(method='first', ascending=False), 10, labels=range(1, 11))
t = ev.groupby('dec', observed=True).agg(n=('y', 'size'), realized=('y', 'mean'), pred=('p', 'mean'))
tb = ev.groupby('decb', observed=True).y.mean()
t['baseline_realized'] = tb.values
out['deciles'] = {int(k): {c: round(float(v), 4) for c, v in r.items()} for k, r in t.iterrows()}
# 상위 20% 포착률
for q in (0.1, 0.2, 0.3):
    k = int(q * len(ev))
    out[f'capture_top{int(q*100)}'] = dict(model=float(ev.y.values[np.argsort(-ev.p.values)[:k]].sum() / ev.y.sum()),
                                          baseline=float(ev.y.values[np.argsort(-ev.pb.values)[:k]].sum() / ev.y.sum()))
# 대형 출원인
cl = np.where(ev.acd.values == 'NA', 'P' + ev.pid.astype(str).values, ev.acd.values)
ev['cl'] = cl
cs = ev.groupby('cl').agg(n=('y', 'size'), rate=('y', 'mean'), p=('p', 'mean')).sort_values('n', ascending=False)
out['largest_clusters'] = [dict(rank=i + 1, n=int(r.n), share=round(float(r.n / len(ev)), 4), realized=round(float(r.rate), 4), pred=round(float(r.p), 4))
                           for i, (_, r) in enumerate(cs.head(10).iterrows())]
top1 = cs.index[0]
m = ev.cl != top1
out['auc_excl_largest'] = dict(model=float(roc_auc_score(ev.y[m], ev.p[m])), baseline=float(roc_auc_score(ev.y[m], ev.pb[m])))
big = ev.cl.isin(cs.index[:10])
out['auc_top10_applicants_only'] = dict(n=int(big.sum()), model=float(roc_auc_score(ev.y[big], ev.p[big])), baseline=float(roc_auc_score(ev.y[big], ev.pb[big])))
out['auc_excl_top10_applicants'] = dict(model=float(roc_auc_score(ev.y[~big], ev.p[~big])), baseline=float(roc_auc_score(ev.y[~big], ev.pb[~big])))
# 산업 수준
g = ev.groupby('ksic').agg(n=('y', 'size'), realized=('y', 'mean'), pred=('p', 'mean'), base=('pb', 'mean'))
out['industry'] = dict(spearman_model=float(spearmanr(g.pred, g.realized)[0]), spearman_base=float(spearmanr(g.base, g.realized)[0]),
                       mae_model=float((g.pred - g.realized).abs().mean()), mae_base=float((g.base - g.realized).abs().mean()),
                       mae_model_meanshift=float((g.pred - g.pred.mean() + g.realized.mean() - g.realized).abs().mean()),
                       mae_base_meanshift=float((g.base - g.base.mean() + g.realized.mean() - g.realized).abs().mean()),
                       top10_overlap_model=len(set(g.pred.nlargest(10).index) & set(g.realized.nlargest(10).index)),
                       top10_overlap_base=len(set(g.base.nlargest(10).index) & set(g.realized.nlargest(10).index)))
# 산업 안 순위(특허 단위 AUC를 KSIC별로)
wa = []
for k, gg in ev.groupby('ksic'):
    if gg.y.nunique() > 1 and len(gg) >= 500:
        wa.append((len(gg), roc_auc_score(gg.y, gg.p), roc_auc_score(gg.y, gg.pb)))
wa = np.array(wa)
out['within_ksic_auc_weighted'] = dict(model=float(np.average(wa[:, 1], weights=wa[:, 0])), baseline=float(np.average(wa[:, 2], weights=wa[:, 0])),
                                       model_better_n=int((wa[:, 1] > wa[:, 2]).sum()), n_ksic=int(len(wa)))
tab = g.assign(rank_real=g.realized.rank(ascending=False), rank_pred=g.pred.rank(ascending=False), rank_base=g.base.rank(ascending=False)).sort_values('realized', ascending=False)
tab.round(4).to_csv(os.path.join(R2, 'audit', 'final_industry_table.csv'), encoding='utf-8-sig')
json.dump(out, open(os.path.join(R2, 'audit', 'diag_final.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print(tab.head(10).round(3).to_string())

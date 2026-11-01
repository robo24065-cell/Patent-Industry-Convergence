"""3단계 독립 대조: final_run.py 코드를 쓰지 않고 cache_holdout/panel_ccy.parquet 에서 직접
레이블(y, breakout)·기준선(kr_mom 로지스틱)·AUC·군집 부트스트랩 CI 를 다시 계산해 holdout_result.json 과 비교."""
import os, json, hashlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
P = pd.read_parquet('cache_holdout/panel_ccy.parquet')
P['cell'] = P.ksic + '|' + P.mg
kr = P[P.cc == 'KR'].pivot_table(index='cell', columns='year', values='n', aggfunc='sum').reindex(columns=range(2000, 2022)).fillna(0)
tot = P.pivot_table(index='cell', columns='year', values='n', aggfunc='sum').reindex(columns=range(2000, 2022)).fillna(0)
kr = kr.reindex(tot.index).fillna(0)
S = lambda df, a, b: df.loc[:, a:b].sum(1)

def frame(t0, label=True):
    d = pd.DataFrame(index=tot.index)
    d['tot_W'] = S(tot, t0 - 2, t0); d['kr_W'] = S(kr, t0 - 2, t0)
    d['kr_mom'] = np.log((d.kr_W + 1) / (S(kr, t0 - 5, t0 - 3) + 1))
    d = d[d.tot_W >= 50].copy(); d['t0'] = t0
    if label:
        d['kr_fut'] = S(kr, t0 + 1, t0 + 3).reindex(d.index)
        d['y'] = np.log((d.kr_fut + 1) / (d.kr_W + 1))
        k = max(1, int(round(0.1 * len(d))))
        key = np.array([int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in d.index])
        o = np.lexsort((key, -(d.kr_fut - d.kr_W).values, -d.y.values))
        yb = np.zeros(len(d), int); yb[o[:k]] = 1; d['yb'] = yb
    return d

te = frame(2018)
tr = pd.concat([frame(t) for t in range(2005, 2016)])
assert (tr.t0 + 3).max() <= 2018
bl = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values)
p0_ind = bl.predict_proba(te[['kr_mom']].values)[:, 1]

R = json.load(open('holdout_result.json', encoding='utf-8'))
pr = pd.read_parquet('preds/HOLDOUT_t0_2018.parquet').set_index('cell')
m = te.merge(pr[['p', 'p_base', 'yb', 'y', 'mg']], left_index=True, right_index=True, suffixes=('', '_fr'), how='outer', indicator=True)
out = dict(n_indep=int(len(te)), n_final=int(len(pr)), merge=m._merge.value_counts().to_dict())
m = m[m._merge == 'both']
out['label_yb_mismatch'] = int((m.yb != m.yb_fr).sum()); out['label_y_maxdiff'] = float((m.y - m.y_fr).abs().max())
out['baseline_coef'] = float(bl.coef_[0, 0])
out['auc_model_recalc'] = float(roc_auc_score(m.yb, m.p))
out['auc_baseline_final'] = float(roc_auc_score(m.yb, m.p_base))
p0i = pd.Series(p0_ind, index=te.index).reindex(m.index)
out['auc_baseline_indep'] = float(roc_auc_score(m.yb, p0i))
out['auc_raw_kr_mom'] = float(roc_auc_score(m.yb, m.kr_mom))
out['skill_recalc'] = (out['auc_model_recalc'] - out['auc_baseline_indep']) / (1 - out['auc_baseline_indep'])
out['prauc'] = float(average_precision_score(m.yb, m.p)); out['base_rate'] = float(m.yb.mean())
k = max(1, int(round(0.1 * len(m)))); out['p_at10'] = float(m.yb.values[np.argsort(-m.p.values)[:k]].mean())
out['rho'] = float(spearmanr(m.y, m.p).statistic)
# 독립 군집 부트스트랩(다른 seed, 메인그룹)
rng = np.random.default_rng(12345); mg = m.mg.values; um = np.unique(mg)
gi = pd.Series(np.arange(len(m))).groupby(mg).apply(lambda s: s.values).to_dict()
yb, p, pb = m.yb.values, m.p.values, p0i.values; bs = []
for _ in range(2000):
    ii = np.concatenate([gi[u] for u in rng.choice(um, len(um))])
    a, a0 = roc_auc_score(yb[ii], p[ii]), roc_auc_score(yb[ii], pb[ii]); bs.append((a, a0, (a - a0) / (1 - a0)))
bs = np.array(bs)
out['ci95_indep'] = {n: [float(x) for x in np.percentile(bs[:, j], [2.5, 97.5])] for j, n in enumerate(['auc', 'auc_baseline', 'skill'])}
out['n_clusters'] = int(len(um))
# 참고: 셀 수가 많은 메인그룹 영향 — 메인그룹 1/m 가중 AUC
w = 1 / pd.Series(mg).map(pd.Series(mg).value_counts()).values
out['auc_mg_weighted'] = float(roc_auc_score(yb, p, sample_weight=w))
out['auc_baseline_mg_weighted'] = float(roc_auc_score(yb, pb, sample_weight=w))
# 규모별
for lo, hi in [(0, 0), (1, 4), (5, 19), (20, 10**9)]:
    s = (m.kr_W >= lo) & (m.kr_W <= hi)
    if s.sum() and m.yb[s].nunique() == 2:
        out[f'auc_krW_{lo}_{hi}'] = dict(n=int(s.sum()), evt=int(m.yb[s].sum()), model=float(roc_auc_score(m.yb[s], m.p[s])),
                                         base=float(roc_auc_score(m.yb[s], p0i[s])))
out['final_json_summary'] = R['summary']; out['final_json_ci'] = R['ci95']
json.dump(out, open('verify_holdout.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1))

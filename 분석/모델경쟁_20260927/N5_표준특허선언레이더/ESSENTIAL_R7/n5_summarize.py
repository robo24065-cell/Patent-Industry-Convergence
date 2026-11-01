# -*- coding: utf-8 -*-
"""R7 N5 결과 집계 → essential_table.csv. 같은 단위·레이블 시험은 full 대비 평균 AUC·skill 차이의 출원인 군집 부트스트랩 95% CI(500회, 원점 블록 함께)."""
import os, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
F = json.load(open(os.path.join(R, 'full.json'), encoding='utf-8')); Fp = pd.read_parquet(os.path.join(R, 'full_preds.parquet'))


def paired(alt, n=500, seed=0):
    m = Fp.merge(alt[['appl_no', 't0', 'p']], on=['appl_no', 't0'], suffixes=('', '_alt'), validate='1:1')
    rng = np.random.default_rng(seed); apps = m.app.unique(); blocks = [g for _, g in m.groupby('t0')]
    idx = [pd.Series(np.arange(len(g))).groupby(g.app.values).apply(lambda s: s.values).to_dict() for g in blocks]
    da, ds = [], []
    for _ in range(n):
        s = rng.choice(apps, len(apps)); a_, s_ = [], []
        for g, ix in zip(blocks, idx):
            ii = np.concatenate([ix[u] for u in s if u in ix]); y = g.y.values[ii]
            if y.min() == y.max(): continue
            af, aa, ab = roc_auc_score(y, g.p.values[ii]), roc_auc_score(y, g.p_alt.values[ii]), roc_auc_score(y, g.p_base.values[ii])
            a_.append(aa - af); s_.append((aa - ab) / (1 - ab) - (af - ab) / (1 - ab))
        da.append(np.mean(a_)); ds.append(np.mean(s_))
    q = lambda v: f'[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]'
    return q(da), q(ds)


rows = []
for f in sorted(glob.glob(os.path.join(R, '*.json'))):
    J = json.load(open(f, encoding='utf-8'))
    if 'summary' not in J: continue
    s = J['summary']; t = J['test']
    r = dict(test=t, auc=round(s['auc'], 4), auc_M0=round(s['auc_M0'], 4), skill=round(s['skill'], 4), d_auc=round(s['auc'] - F['summary']['auc'], 4),
             d_skill=round(s['skill'] - F['summary']['skill'], 4), top100=round(s['top100'], 1), top100_M0=round(s['top100_M0'], 1),
             n=round(s['n'], 0), pos=round(s['pos'], 1), wins_vs_M0=f"{s['wins_vs_M0']}/7",
             auc_lt_full=f"{sum(J['by_origin'][o]['auc'] < F['by_origin'][o]['auc'] for o in F['by_origin'])}/7" if t not in ('conc_task', 'concm_task') else '')
    if 'skill_vs_M0_alt' in s: r['skill_vs_M0_alt'] = round(s['skill_vs_M0_alt'], 4)
    if t not in ('full', 'conc_task', 'concm_task'):
        r['d_auc_ci95'], r['d_skill_ci95'] = paired(pd.read_parquet(f.replace('.json', '_preds.parquet')))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'essential_table.csv'), index=False, encoding='utf-8-sig')

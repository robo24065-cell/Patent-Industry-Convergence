# -*- coding: utf-8 -*-
"""R7 T3 결과 집계: results/*.json → essential_table.csv. 목표 행이 같은 시험(제거·KR 재계산·결과 고정 대체)은
full 대비 평균 AUC 차이의 IPC 메인그룹 군집 부트스트랩 95% CI(500회, 원점 블록 함께 재표집)를 붙인다."""
import os, json, glob, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')


def paired_boot(full, alt, n=500, seed=0):
    m = full.merge(alt[['cell', 't0', 'p']], on=['cell', 't0'], suffixes=('', '_alt'), validate='1:1')
    rng = np.random.default_rng(seed); mgs = m.mg.unique()
    blocks = [g for _, g in m.groupby('t0')]
    idx = [pd.Series(np.arange(len(g))).groupby(g.mg.values).apply(lambda s: s.values).to_dict() for g in blocks]
    out = []
    for _ in range(n):
        s = rng.choice(mgs, len(mgs)); d = []
        for g, ix in zip(blocks, idx):
            ii = np.concatenate([ix[u] for u in s if u in ix]); y = g.yb.values[ii]
            if y.min() == y.max(): continue
            d.append(roc_auc_score(y, g.p_alt.values[ii]) - roc_auc_score(y, g.p.values[ii]))
        out.append(np.mean(d))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


rows = []
VARS = sys.argv[1:] or ['pub', 'app', 'pubF']   # 9분 제한 때문에 변형별로 나눠 실행 → essential_table_{var}.csv, 마지막에 합침
for var in VARS:
    fj = os.path.join(R, f'{var}_full.json')
    if not os.path.exists(fj): continue
    F = json.load(open(fj, encoding='utf-8')); Fp = pd.read_parquet(os.path.join(R, f'{var}_full_preds.parquet'))
    for f in sorted(glob.glob(os.path.join(R, f'{var}_*.json'))):
        J = json.load(open(f, encoding='utf-8')); s = J['summary']; test = J['test']
        if J['var'] != var: continue   # 'pub_*' 글롭이 'pubF_*' 도 잡으므로
        r = dict(var=var, test=test, mean_auc=round(s['mean_auc'], 4), mean_base_auc=round(s['mean_base_auc'], 4), mean_skill=round(s['mean_skill'], 4),
                 d_auc=round(s['mean_auc'] - F['summary']['mean_auc'], 4), d_skill=round(s['mean_skill'] - F['summary']['mean_skill'], 4),
                 mean_p10=round(s['mean_p10'], 4), n_cells=round(s['mean_n'], 1), n_evt=round(s['mean_evt'], 1),
                 auc_lt_full=f"{sum(J['by_origin'][o]['model']['auc'] < F['by_origin'][o]['model']['auc'] for o in F['by_origin'] if o in J['by_origin'])}/{len(F['by_origin'])}",
                 skill_pos=f"{sum(J['by_origin'][o]['skill'] > 0 for o in J['by_origin'])}/{len(J['by_origin'])}")
        if test not in ('full', 'conc_task', 'concm_task'):
            ci = paired_boot(Fp, pd.read_parquet(f.replace('.json', '_preds.parquet'))); r['d_auc_ci95'] = f'[{ci[0]:.4f}, {ci[1]:.4f}]'
        cov = J.get('info', {}).get('coverage_D')
        if cov: r['alt_cell_nonzero'] = round(cov['alt_totW_pos'], 3)
        rows.append(r); print(r, flush=True)
T = pd.DataFrame(rows)
for var in VARS: T[T['var'] == var].to_csv(os.path.join(HERE, f'essential_table_{var}.csv'), index=False, encoding='utf-8-sig')
parts = [pd.read_csv(os.path.join(HERE, f'essential_table_{v}.csv')) for v in ['pub', 'app', 'pubF'] if os.path.exists(os.path.join(HERE, f'essential_table_{v}.csv'))]
pd.concat(parts, ignore_index=True).to_csv(os.path.join(HERE, 'essential_table.csv'), index=False, encoding='utf-8-sig')

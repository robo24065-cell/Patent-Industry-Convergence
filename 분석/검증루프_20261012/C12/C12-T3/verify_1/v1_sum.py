# -*- coding: utf-8 -*-
import os, sys, json, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = os.path.dirname(os.path.abspath(__file__))
WD = os.path.dirname(OUT)
rows = []
for k in range(5):
    r = json.load(open(os.path.join(OUT, f'v1_final_s{k}.json'), encoding='utf-8'))
    for e, v in r.items(): rows.append(dict(part='final2021', seed=k, exp=e, skill=v['skill'], auc=v['auc'], hits=v['hits_top504']))
for fn in sorted(glob.glob(os.path.join(OUT, 'v1_dev_s*.json'))):
    k = int(fn[-6]); r = json.load(open(fn, encoding='utf-8'))
    for e, d in r.items():
        if len(d) == 9: rows.append(dict(part='dev2010_18', seed=k, exp=e, skill=float(np.mean([x['skill'] for x in d.values()])), auc=float(np.mean([x['auc'] for x in d.values()])), hits=np.nan,
                                         by_origin=json.dumps({o: round(x['skill'], 6) for o, x in d.items()})))
df = pd.DataFrame(rows)
full = df[df.exp == 'full'][['part', 'seed', 'skill']].rename(columns={'skill': 'skill_full'})
df = df.merge(full, on=['part', 'seed']); df['d_vs_full'] = df.skill - df.skill_full
df.to_csv(os.path.join(OUT, 'v1_results.csv'), index=False, encoding='utf-8-sig')
summ = df.groupby(['part', 'exp']).agg(n_seed=('seed', 'nunique'), skill_mean=('skill', 'mean'), d_mean=('d_vs_full', 'mean'), d_sd=('d_vs_full', 'std'),
                                         d_pos=('d_vs_full', lambda s: int((s > 0).sum())), d_min=('d_vs_full', 'min'), d_max=('d_vs_full', 'max')).reset_index()
print(summ.to_string())
summ.to_csv(os.path.join(OUT, 'v1_summary.csv'), index=False, encoding='utf-8-sig')
# 작업 b_sens 대조
b = pd.read_csv(os.path.join(WD, 'b_sens.csv'))
b = b[b.seed.astype(str).str.isdigit()]; b['seed'] = b.seed.astype(int)
for sc, e in [('S0', 'full'), ('S3', 'no_pi')]:
    m = b[b.scenario == sc].merge(df[(df.part == 'final2021') & (df.exp == e)], on='seed')
    print(sc, e, 'maxdiff vs b_sens', float(np.abs(m.skill - m.auc_x * 0 - m.skill).max()) if False else float(np.abs(m['skill_x'] - m['skill_y']).max()))
# 원점별 dev 차(seed 0)
d0 = df[(df.part == 'dev2010_18')]
for k in sorted(d0.seed.unique()):
    f = json.loads(d0[(d0.seed == k) & (d0.exp == 'full')].by_origin.iloc[0])
    for e in ['no_pi', 'no_ros']:
        q = d0[(d0.seed == k) & (d0.exp == e)]
        if len(q):
            g = json.loads(q.by_origin.iloc[0]); dd = {o: round(g[o] - f[o], 4) for o in f}
            print('dev seed', k, e, 'by-origin Δ', dd, 'wins(제거가 나음)', sum(v > 0 for v in dd.values()), '/9')

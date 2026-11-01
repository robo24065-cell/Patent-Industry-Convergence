# -*- coding: utf-8 -*-
"""트렌드 산출물(개발 원점, 표본 밖 예측): KSIC × 기준일별 예측 포기위험·기준선·실현 3년 불납률 → R2/dev_ksic_risk_trend.csv
사용: python make_trend_dev.py <챔피언 exp_id>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import r2lib as R

exp = sys.argv[1]
spec = json.load(open(os.path.join(R.R2, 'specs', f'{exp}.json'), encoding='utf-8'))['spec']
nm = pd.read_csv(R.SPEC0['paths']['ksic_names'], dtype={'세세분류코드': str}).drop_duplicates('세세분류코드').set_index('세세분류코드')['산업명']
rows, sp = [], {}
for o, C, EL in R.ORIGINS:
    ev = R.ev_meta(o).copy()
    ev['p'], _ = R.ensemble(o, spec)
    ev['p_base'] = R.fit_predict_sk(o, spec, 'base')
    g = ev.groupby(['L', 'ksic']).agg(n_alive=('pid', 'size'), pred_risk=('p', 'mean'), base_risk=('p_base', 'mean'), realized_rate=('y', 'mean')).reset_index()
    g['origin'] = o; g['C_train_label_cutoff'] = C
    rows.append(g)
    for L, gg in g.groupby('L'):
        sp[L] = dict(model=round(float(spearmanr(gg.pred_risk, gg.realized_rate)[0]), 3), baseline=round(float(spearmanr(gg.base_risk, gg.realized_rate)[0]), 3))
d = pd.concat(rows, ignore_index=True)
d['industry'] = d.ksic.map(nm)
d['rank_pred'] = d.groupby('L').pred_risk.rank(ascending=False)
d['rank_real'] = d.groupby('L').realized_rate.rank(ascending=False)
d.to_csv(os.path.join(R.R2, 'dev_ksic_risk_trend.csv'), index=False, encoding='utf-8-sig')
json.dump(dict(exp=exp, industry_spearman=sp), open(os.path.join(R.R2, 'dev_industry_validity.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(sp, indent=1))
last = d[d.L == '2018-07-01'].sort_values('pred_risk', ascending=False)
print(last.head(8)[['ksic', 'industry', 'n_alive', 'pred_risk', 'realized_rate']].to_string())

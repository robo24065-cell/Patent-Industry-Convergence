# -*- coding: utf-8 -*-
# 산업 순위 ρ 독립 재계산(pandas groupby + scipy) + 지속 지수 2024 창 사건 수 SQL 재집계(다른 경로)
import os, json, sqlite3, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
fz = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet')); lb = pd.read_parquet(os.path.join(W, 'cache', 'labels.parquet'))
pi = pd.read_csv(os.path.join(W, 'persist_index.csv'), dtype={'ksic': str}).set_index('ksic')
d = pd.DataFrame({'k': fz.ksic.astype(str), 'p': fz.p, 'y': lb.y, 'b2': fz.big2})
g = d.groupby('k').agg(p=('p', 'mean'), y=('y', 'mean'))
g2 = d[d.b2 == 0].groupby('k').agg(p=('p', 'mean'), y=('y', 'mean'))
O = dict(rho_model=spearmanr(g.p, g.y)[0], rho_persist=spearmanr(pi.persist_rate.reindex(g.index), g.y)[0],
         rho_model_excl_big2=spearmanr(g2.p, g2.y)[0], rho_persist_excl_big2=spearmanr(pi.persist_rate.reindex(g2.index), g2.y)[0])
r = json.load(open(os.path.join(W, 'prospective_result.json'), encoding='utf-8'))['industry']['stats']
O['diff'] = {k: abs(O[k] - r[k]['est']) for k in ['rho_model', 'rho_persist', 'rho_model_excl_big2', 'rho_persist_excl_big2']}
# 2위 출원인을 뺀 뒤 지수 1위 산업과 실현
O['top5_index'] = g.sort_values('p', ascending=False).head(5).round(4).reset_index().values.tolist()
O['top5_index_excl_big2'] = g2.sort_values('p', ascending=False).head(5).round(4).reset_index().values.tolist()
json.dump(O, open(os.path.join(HERE, 'v0b_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))

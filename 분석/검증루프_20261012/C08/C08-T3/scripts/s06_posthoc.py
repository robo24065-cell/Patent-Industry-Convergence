# -*- coding: utf-8 -*-
# 사후 진단(비사전등록, 판정 미사용): 26211·26212 실현율을 좌우한 출원인 몫, 상위 3개 대형 출원인 제외 산업 ρ, 3위 출원인 과거 같은 창 사건
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import c08lib as C
fz = pd.read_parquet(os.path.join(C.W, 'frozen_scores.parquet')); lb = pd.read_parquet(os.path.join(C.CACHE, 'labels.parquet'))
pi = pd.read_csv(os.path.join(C.W, 'persist_index.csv'), dtype={'ksic': str}).set_index('ksic')
d = fz[['pid', 'ksic', 'acd', 'p', 'due_win']].assign(y=lb.y.values, ksic=fz.ksic.astype(str))
vc = d[d.acd != 'NA'].acd.value_counts(); top = vc.index[:3].tolist()
O = {}
for k in ('26211', '26212'):
    s = d[d.ksic == k]
    g = s.groupby('acd').agg(n=('y', 'size'), ev=('y', 'sum'), mean_p=('p', 'mean')).sort_values('n', ascending=False).head(4)
    g['rank_overall'] = [int(np.where(vc.index == a)[0][0]) + 1 if a in vc.index else None for a in g.index]
    O[k] = dict(n=len(s), events=int(s.y.sum()), top_applicants=g.reset_index().drop(columns='acd').round(4).to_dict('records'))
for nm, ex in (('excl_top3', top), ('excl_rank3', [top[2]]), ('excl_rank2_3', top[1:3])):
    s = d[~d.acd.isin(ex)]; g = s.groupby('ksic').agg(p=('p', 'mean'), y=('y', 'mean'))
    O['rho_' + nm] = dict(model=spearmanr(g.p, g.y)[0], persist=spearmanr(pi.persist_rate.reindex(g.index), g.y)[0])
D = C.load(); lap = C.lapse_events(D); ap = D['pt'].set_index('pid').acd
for yy in (2021, 2022, 2023, 2024, 2025):
    Ly = np.datetime64(f'{yy}-07-01'); Wy = np.datetime64(f'{yy}-10-31')
    r_, _, _ = C.risk_set(D, Ly); r3 = r_[ap.reindex(r_).values == top[2]]
    ee = lap[(lap.E > Ly) & (lap.E <= Wy) & (lap.cd <= np.datetime64('2026-08-15'))].pid.unique()
    O[f'rank3_window_{yy}'] = dict(riskset=len(r3), events=int(np.isin(r3, ee).sum()))
O['rank3_share_of_all_pos'] = float(d[d.acd == top[2]].y.sum() / d.y.sum())
C.jdump(O, os.path.join(C.CACHE, 's06_posthoc.json')); print(json.dumps(O, ensure_ascii=False, indent=1, default=str))

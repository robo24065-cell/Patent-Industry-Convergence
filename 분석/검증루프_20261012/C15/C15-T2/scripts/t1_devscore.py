# -*- coding: utf-8 -*-
"""T1 개발 원점 채점(동결 final_run_r3 --mode dev 채점과 같은 계산): 원점별 skill = skill(mets(y_u, blend, newc)['auc_all'], baselines(c)['strongest']['auc_all']).
기준선은 공식 고정(R3/cache/o{c}/baselines_r3.json, 읽기만). b=0은 공식 FROZEN_dev, b≥1은 cache/t1/b{b}/dev.
 python t1_devscore.py <b>  → cache/t1/b{b}_dev.json"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from t1common import *
import numpy as np, pandas as pd
import final_run_r3 as F, r3
from r3 import mets, skill

b = int(sys.argv[1]); rows = {}
for c in r3.ORIGINS:
    f = (R3/'runs'/'FROZEN_dev'/f'o{c}.parquet') if b == 0 else (rd(b, c)/f'o{c}.parquet')
    o = pd.read_parquet(f); o = o[~o.prior_any.values]; s = F.blend(o)
    m = mets(o.y_u.values, s, o.newc.values); bb = json.load(open(R3/'cache'/f'o{c}'/'baselines_r3.json', encoding='utf-8'))['strongest']
    rows[str(c)] = dict(auc=m['auc_all'], base_auc=bb['auc_all'], skill=skill(m['auc_all'], bb['auc_all']), top1000=int(o.y_u.values[np.argsort(-s)[:1000]].sum()))
out = dict(bundle=b, seeds=seeds(b), per_origin=rows, mean_skill=float(np.mean([r['skill'] for r in rows.values()])))
if b == 0:
    off = json.load(open(R3/'dev_result_r3.json', encoding='utf-8'))
    out['official_mean_skill'] = off['mean_skill']; out['d_vs_official'] = out['mean_skill'] - off['mean_skill']
json.dump(out, open(CT1/f'b{b}_dev.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out)

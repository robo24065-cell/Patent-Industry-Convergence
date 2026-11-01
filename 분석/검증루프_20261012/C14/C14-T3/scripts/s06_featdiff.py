# -*- coding: utf-8 -*-
"""s06: 변형별 피처 변화 크기(원점 2019 평가 패널, 학습 원점 2012) — inj/parts 비교 → cache/s06_featdiff.json"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd
P = HERE/'inj'/'parts'; OUT = {}
def cmp(a, b, cols):
    r = {}
    for x in cols:
        u = a[x].values.astype(np.float64); v = b[x].values.astype(np.float64)
        m = np.isfinite(u) & np.isfinite(v)
        r[x] = dict(changed_share=float((u[m] != v[m]).mean()), mean_a=float(u[m].mean()), mean_b=float(v[m].mean()),
                    corr=float(np.corrcoef(u[m], v[m])[0, 1]) if u[m].std() > 0 and v[m].std() > 0 else None,
                    mean_abs_diff=float(np.abs(u[m] - v[m]).mean()))
    return r
for c, t in ((2019, 2019), (2019, 2012), (2008, 2008)):
    p0 = pd.read_parquet(P/f'P0_o{c}_c{t}.parquet'); key = f'o{c}_c{t}'
    main = ['lout', 'lsize', 'inshare', 't_lsum', 't_mean', 't_max', 'ot_mean', 'ot_max', 'ot_top3', 'ot_hi', 'oin50', 'now3', 'gap_last']
    main = [x for x in main if x in p0.columns]
    for v in ('S2', 'S3_1', 'S4'):
        OUT[f'{key}_{v}_vs_P0'] = cmp(p0, pd.read_parquet(P/f'{v}_o{c}_c{t}.parquet'), main)
    a = pd.read_parquet(P/f'AN_A_o{c}_c{t}.parquet'); b = pd.read_parquet(P/f'AN_B_o{c}_c{t}.parquet')
    OUT[f'{key}_ANB_vs_ANA'] = cmp(a, b, list(a.columns))
    OUT[f'{key}_ANA_vs_P0_ot'] = cmp(p0, a, list(a.columns))
    ti = pd.read_parquet(P/f'AN_TI_o{c}_c{t}.parquet')
    OUT[f'{key}_ANTI_cols'] = list(ti.columns)
json.dump(OUT, open(HERE/'cache/s06_featdiff.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in OUT.items():
    if isinstance(v, dict):
        print(k, {x: (round(y['changed_share'], 3), round(y['mean_a'], 4), round(y['mean_b'], 4), None if y['corr'] is None else round(y['corr'], 3)) for x, y in v.items()})

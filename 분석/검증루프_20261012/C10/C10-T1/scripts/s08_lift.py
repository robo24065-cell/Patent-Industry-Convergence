# -*- coding: utf-8 -*-
"""s08: '목표 산업 디자인(확률>0.2) 보유 쌍의 진입률 배수'(ext_results_A: c=2012 5.35배, 실용신안 4.66배)를 S0·S1·S2 입력으로 다시 계산.
 원점 2012(개발, 4년창 레이블 = prep_te.y_u)와 원점 2019(최종창 2020–23, cache/final_labels). 진단 전용 → cache/s08_lift.json"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
R3 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'
sys.argv = ['x', 'S0', '2019']
import run_var as RV
out = {}
for c, FC in ((2012, R3/'cache'/'o2012'), (2019, R3/'cache_final'/'o2019')):
    if c == 2019:
        te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut'])
        L = pd.read_parquet(HERE/'cache/final_labels.parquet'); te['y'] = L.y_u.values; te['pa'] = L.prior_any.values
    else:
        te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut', 'y_u', 'prior_any']).rename(columns={'y_u': 'y', 'prior_any': 'pa'})
    te = te[~te.pa.values].reset_index(drop=True)
    tr0 = te.iloc[:0].copy()
    RV.r3ext.PDIR[c] = FC
    for kind, pre in (('design', 'ds'), ('um', 'um')):
        for var in ('S0', 'S1', 'S2'):
            if var == 'S0':
                _, t = RV.r3ext._title_exposure(tr0.copy(), te[['firm', 'k', 'cut']].copy(), c, kind, pre)
            else:
                _, t = RV.title_exposure_vf(tr0.copy(), te[['firm', 'k', 'cut']].copy(), c, kind, pre, FC, var)
            m = t[f'{pre}_max'].values > 0.2; y = te.y.values
            has = np.expm1(t[f'{pre}_n'].values) > 0
            out[f'o{c}_{kind}_{var}'] = dict(pairs=int(len(y)), share_hi=float(m.mean()), rate_hi=float(y[m].mean()), rate_all=float(y.mean()),
                                            lift=float(y[m].mean() / y.mean()), firm_has_share=float(has.mean()), lift_has_vs_not=float(y[has].mean() / y[~has].mean()))
            print(c, kind, var, {k: round(v, 4) for k, v in out[f'o{c}_{kind}_{var}'].items()}, flush=True)
json.dump(out, open(HERE/'cache/s08_lift.json', 'w', encoding='utf-8'), indent=1)

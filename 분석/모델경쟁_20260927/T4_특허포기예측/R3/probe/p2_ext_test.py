# -*- coding: utf-8 -*-
"""새 피처 시험 계산(O18 평가 프레임) — 결측률·평가 레이블과의 단순 관계(개발 원점 O18 = 결과 창 <= 2021-07)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r3lib, r3ext, numpy as np, pandas as pd
R = r3lib.R
t = time.time()
for o in ['O16', 'O18']:
    d = R._read_frame(o, 'ev.parquet', ['pid', 'L', 'ksic', 'next_annl', 'acd', 'age', 'atype', 'y'])
    cols = ['sep_self', 'sep_fam', 'sep_any', 'sep_a_share', 'np_pres', 'np_act', 'np_lemp', 'np_lossr', 'np_gemp', 'np_exit',
            'kp_n', 'kp_lxi', 'kp_has', 'pr_ppi_g1', 'pr_ppi_g3', 'pr_exp_g3', 'ap_late1y', 'ap_late_d', 'ap_multi1y', 'ap_n1y', 'ap_lapacc']
    d = r3ext.add(d, cols)
    print(o, len(d), 'rate', round(d.y.mean(), 4), f'{time.time()-t:.0f}s', 'peak', R.peak_mem_gb())
    for c in cols:
        v = d[c]
        s = f'{c:14s} nan {v.isna().mean():.3f} mean {v.mean():.4f}'
        if v.nunique() <= 2:
            s += ' | y|1 ' + str(round(d.y[v == 1].mean(), 4)) + f' (n={int((v==1).sum())}) y|0 ' + str(round(d.y[v == 0].mean(), 4))
        else:
            ok = v.notna()
            if ok.sum() > 1000:
                q = pd.qcut(v[ok].rank(method='first'), 5, labels=False)
                s += ' | 5분위 y ' + str(d.y[ok].groupby(q.values).mean().round(3).tolist())
        print(s, flush=True)

# -*- coding: utf-8 -*-
"""단위 점검: run_var.title_exposure_vf 에 valid_from = 등록일(제거 없음)을 주면 동결 r3ext._title_exposure 와 값이 같아야 한다.
 + S1·S2 valid_from 에서 원점 2019 행 제거 수가 s04 통계와 같은지. → cache/t_equiv.json"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
R3 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'
sys.argv = ['x', 'S0', '2019']
import run_var as RV          # 쓰기 가드 포함, F·r3ext import
FC = R3/'cache_final'/'o2019'
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut']).iloc[::7].reset_index(drop=True)
tr = pd.read_parquet(FC/'prep_tr.parquet', columns=['firm', 'k', 'cut']).iloc[::11].reset_index(drop=True)
out = {}
for kind, pre in (('design', 'ds'), ('um', 'um')):
    ix = pd.read_parquet(FC/f'P_{kind}_index.parquet'); rg = ix.rgst_date.astype(str).str[:8].astype(np.int64).values
    np.save(HERE/'cache'/f'vf_TEQ_{kind}_o2019.npy', rg)
    a_tr, a_te = RV.r3ext._title_exposure(tr.copy(), te.copy(), 2019, kind, pre) if False else (None, None)
    RV.r3ext.PDIR[2019] = FC
    a_tr, a_te = RV.r3ext._title_exposure(tr.copy(), te.copy(), 2019, kind, pre)
    b_tr, b_te = RV.title_exposure_vf(tr.copy(), te.copy(), 2019, kind, pre, FC, 'TEQ')
    cols = [f'{pre}_lsum', f'{pre}_max', f'{pre}_n20', f'{pre}_rec3', f'{pre}_n']
    out[kind] = dict(max_abs_te=float(np.nanmax(np.abs(a_te[cols].values - b_te[cols].values))), max_abs_tr=float(np.nanmax(np.abs(a_tr[cols].values - b_tr[cols].values))))
    for var in ('S1', 'S2'):
        vf = np.load(HERE/'cache'/f'vf_{var}_{kind}_o2019.npy')
        c_tr, c_te = RV.title_exposure_vf(tr.copy(), te.copy(), 2019, kind, pre, FC, var)
        out[kind][f'{var}_removed_rows_cut2019'] = int(((rg <= 20191231) & (vf > 20191231)).sum())
        out[kind][f'{var}_te_mean_lsum_ratio'] = float(c_te[f'{pre}_lsum'].mean() / a_te[f'{pre}_lsum'].mean())
        out[kind][f'{var}_te_share_pairs_changed'] = float((np.abs(c_te[cols].values - a_te[cols].values).max(1) > 1e-6).mean())
    (HERE/'cache'/f'vf_TEQ_{kind}_o2019.npy').unlink()
json.dump(out, open(HERE/'cache/t_equiv.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))

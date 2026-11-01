# -*- coding: utf-8 -*-
"""h01 절단 검사에서 2021·2022 예측이 미세하게 달랐던 원인 진단(학습 행렬·타깃·순위 비교). 모형 변경 없음."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import final_run as fr
spec = fr.load_spec(); ms = fr.model_spec(spec); f = ms['features']
Ffull = fr.build_panel(2024, 2024)
for T in [2021, 2022, 2023]:
    Ft = fr.build_panel(T - 1, T)
    key = ['ksic', 'year']
    Rf = fr.rank_features(Ffull, f, ms['sectors'], True); Rt = fr.rank_features(Ft, f, ms['sectors'], True)
    trf = fr._train_rows(Ffull, T, ms); trt = fr._train_rows(Ft, T, ms)
    a = pd.concat([trf[key].reset_index(drop=True), Rf.loc[trf.index].reset_index(drop=True)], axis=1)
    b = pd.concat([trt[key].reset_index(drop=True), Rt.loc[trt.index].reset_index(drop=True)], axis=1)
    print(T, 'train rows', len(a), len(b), 'same keys/order', (a[key].values == b[key].values).all())
    # 원 피처(순위 전) 비교
    ra = Ffull.loc[trf.index, key + f].reset_index(drop=True); rb = Ft.loc[trt.index, key + f].reset_index(drop=True)
    for c in f:
        d = (ra[c] - rb[c]).abs(); nn = (ra[c].isna() != rb[c].isna()).sum()
        rd = (a[c] - b[c]).abs()
        if d.max() > 1e-12 or nn or rd.max() > 1e-12:
            yrs = sorted(ra.loc[(d > 1e-12) | (ra[c].isna() != rb[c].isna()), 'year'].unique())
            ryrs = sorted(a.loc[(rd > 1e-12) | (a[c].isna() != b[c].isna()), 'year'].unique())
            print('  ', c, 'rawdiff', float(d.max()), 'nan-mismatch', int(nn), 'raw yrs', yrs, '| rankdiff', float(rd.max()), 'rank yrs', ryrs)
    print('  y same', np.allclose(trf.y.values, trt.y.values))
    tf = Ffull[(Ffull.year == T) & (Ffull.in50 == 1)]; tt = Ft[(Ft.year == T) & (Ft.in50 == 1)]
    print('  test raw feat maxdiff', float((tf[f].reset_index(drop=True) - tt[f].reset_index(drop=True)).abs().max().max()),
          'test rank maxdiff', float((Rf.loc[tf.index].reset_index(drop=True) - Rt.loc[tt.index].reset_index(drop=True)).abs().max().max()))

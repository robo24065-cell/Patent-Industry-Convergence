# 1라운드 캐시(전체 소멸 적재)와 2라운드 캐시(E<2022 컷)의 같은 기준일 피처가 동일한지 → seed_cache_from_r1 안전성 확인
import os, pandas as pd, numpy as np
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T4 = os.path.dirname(R2)
for pre in ['feat_', 'soc_', 's2ag_', 's2ow_']:
    for L in ['20160101', '20180101', '20180701']:
        a = os.path.join(T4, 'cache', f'{pre}{L}.parquet'); b = os.path.join(R2, 'cache', f'{pre}{L}.parquet')
        if not (os.path.exists(a) and os.path.exists(b)): print(pre, L, 'missing', os.path.exists(a), os.path.exists(b)); continue
        x = pd.read_parquet(a); y = pd.read_parquet(b)
        same = x.shape == y.shape and list(x.columns) == list(y.columns)
        diff = []
        if same:
            for c in x.columns:
                if not x[c].equals(y[c]):
                    xv, yv = x[c].values, y[c].values
                    try:
                        if np.allclose(xv.astype(float), yv.astype(float), equal_nan=True): continue
                    except Exception: pass
                    diff.append(c)
        print(pre, L, x.shape, y.shape, 'same_shape_cols', same, 'diff_cols', diff)
m = pd.read_csv(r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv', encoding='utf-8-sig', dtype={'연월': str})
print('KOSIS months', m['연월'].min(), m['연월'].max(), m['항목명'].unique()[:6])

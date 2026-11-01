# -*- coding: utf-8 -*-
"""s03: inj/parts/<var>_o<c>_c<t>.parquet → inj/<var>/o<c>_tr.parquet(학습 원점 순서대로 이어붙임, prep_tr 행 순서) · o<c>_te.parquet
 python s03_assemble.py <var> <c>"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard
HERE = guard.HERE
import numpy as np, pandas as pd, pyarrow.parquet as pq
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
R3 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'
v, c = sys.argv[1], int(sys.argv[2])
cuts = (2012, 2013, 2014, 2015) if c == 2019 else tuple(range(c - 7, c - 3))
FCp = R3/'cache_final'/'o2019' if c == 2019 else R3/'cache'/f'o{c}'
P = HERE/'inj'/'parts'; O = HERE/'inj'/v; O.mkdir(parents=True, exist_ok=True)
tr_cut = pd.read_parquet(FCp/'prep_tr.parquet', columns=['cut']).cut.values
parts = []
for t in cuts:
    x = pd.read_parquet(P/f'{v}_o{c}_c{t}.parquet'); n = int((tr_cut == t).sum())
    assert len(x) == n, (t, len(x), n); parts.append(x)
tr = pd.concat(parts, ignore_index=True)
assert len(tr) == len(tr_cut)
# prep_tr 가 원점 순서로 이어붙여졌는지(행 순서) 확인
assert (np.diff(np.searchsorted(np.array(cuts), tr_cut)) >= 0).all(), 'prep_tr 원점 순서가 아님'
te = pd.read_parquet(P/f'{v}_o{c}_c{c}.parquet')
assert len(te) == pq.ParquetFile(FCp/'prep_te.parquet').metadata.num_rows
tr.to_parquet(O/f'o{c}_tr.parquet', index=False); te.to_parquet(O/f'o{c}_te.parquet', index=False)
print('assembled', v, c, tr.shape, te.shape, list(tr.columns)[:6])

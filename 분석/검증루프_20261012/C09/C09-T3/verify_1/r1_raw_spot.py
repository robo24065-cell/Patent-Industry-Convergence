# -*- coding: utf-8 -*-
"""원본 TSV 1개(31202, 최소 크기)에서 parquet 변환 전 출원인명 결측을 직접 확인 (읽기 전용)."""
import pandas as pd, glob, os, sys, io, duckdb
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
f = glob.glob(os.path.join(B, '공모전데이터', '융합데이터_*', '31202.csv'))[0]
cols = ['출원번호', '출원연도', '등록일자', '공개일자', '국가코드', '출원인명']
d = pd.concat(pd.read_csv(f, sep='\t', encoding='utf-8-sig', usecols=cols, dtype=str, chunksize=200_000))
n_all = len(d); null_all = int(d['출원인명'].fillna('').str.strip().eq('').sum())
def yr(s): return pd.to_numeric(s.str[:4], errors='coerce')
d['py'] = pd.concat([yr(d['공개일자']), yr(d['등록일자'])], axis=1).min(axis=1)
d['ay'] = pd.to_numeric(d['출원연도'], errors='coerce')
d = d[d.ay >= 2000].copy()
d['miss'] = d['출원인명'].fillna('').str.strip().str.split('|').str[0].str.strip().eq('')
g = d[d.py.between(2000, 2024)].groupby(['국가코드', 'py']).miss.agg(['size', 'mean']).reset_index()
cc = g['국가코드']
sel = g[((cc == 'US') & g.py.between(2001, 2006)) | ((cc == 'CN') & g.py.between(2005, 2011)) | ((cc == 'JP') & g.py.between(2019, 2023))]
print(sel.round(4).to_string())
for y, k in [(2008, 12), (2018, 8)]:
    s = d[(d['국가코드'] == 'US') & (d.py == y) & ~d.miss]['출원인명'].str.split('|').str[0]
    print(f'US pub{y} sample:', s.sample(min(k, len(s)), random_state=1).tolist())
M = os.path.join(B, '분석', '모델경쟁_20260927', '_shared', 'meta', '31202.parquet').replace(os.sep, '/')
q = duckdb.sql(f"select count(*), sum((applicant is null)::int) from read_parquet('{M}')").fetchone()
print('parquet rows/null applicant', q, '| tsv rows', n_all, 'tsv null applicant', null_all)

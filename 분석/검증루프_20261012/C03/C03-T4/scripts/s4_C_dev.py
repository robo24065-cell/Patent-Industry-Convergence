# s4_C: C03-C 개발 원점 P1 집계
import sys, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from libC import *
T0 = time.time()
d = load()
print('docs rows', len(d), 'firms', d.cd.nunique(), 'ksic in 50', d.ksic.isin(KSIC50).mean())
rows, P = [], []
for t0 in DEV:
    st, pos, r = build(t0, d)
    rows.append(st); P.append(pos); print(st, round(time.time() - T0), flush=True)
R = pd.DataFrame(rows); P = pd.concat(P, ignore_index=True)
R.to_csv(C + r'\s4_C_dev_stats.csv', index=False, encoding='utf-8-sig'); P.to_parquet(C + r'\C_dev_pos.parquet', index=False)
print('origin sum', len(P), 'unique (cd,k,t0)', len(P.drop_duplicates(['cd', 'k', 't0'])), 'unique (cd,k)', len(P.drop_duplicates(['cd', 'k'])), 'min', R.pos.min())

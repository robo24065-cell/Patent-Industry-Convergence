# s2_A: C03-A 개발 원점 P1 집계(주 정의 PUB 연결, 진단 PUB+REG)
import sys, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from libA import *
T0 = time.time()
rows, P = [], []
for meth in [('PUB',), ('PUB', 'REG')]:
    docs, L = load(meth)
    for t0 in DEV:
        st, pos, risk = build(t0, docs, L)
        st['link'] = '+'.join(meth)
        rows.append(st)
        if meth == ('PUB',):
            P.append(pos)
        print(st, round(time.time() - T0), flush=True)
R = pd.DataFrame(rows)
R.to_csv(C + r'\s2_A_dev_stats.csv', index=False, encoding='utf-8-sig')
P = pd.concat(P, ignore_index=True)
P.to_parquet(C + r'\A_dev_pos.parquet', index=False)
u = P.drop_duplicates(['cd', 'k'])
print('PUB origin sum', len(P), 'unique (cd,k)', len(u), 'min per origin', R[R.link == 'PUB'].pos.min())

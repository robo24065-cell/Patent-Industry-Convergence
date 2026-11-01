# 반박 검증 r1: (1) 산술 (2) 원점별 y 합·2023/2024(비잠금) 양성률 (3) 양수인 포트폴리오에 p 자신이 t0 전에 들어가 Y=0이 되는 경우(경로 B/C 자기포함) 점검
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4'
C = W + r'\cache'
print('arith', 337082 * (453/294655 + 397/305597) / 2)
PI = pd.read_parquet(C + r'\PI.parquet')
pat = pd.read_parquet(C + r'\pat.parquet', columns=['rgst_no', 'applNo'])
rows = []
for Y in range(2015, 2025):
    b = pd.read_parquet(C + rf'\base_{Y}.parquet', columns=['rgst_no', 'applNo', 'has_ev', 'buyer', 'buyer_type'])
    l = pd.read_parquet(C + rf'\lab_prov_{Y}.parquet')
    d = b.merge(l, on='rgst_no')
    e = d[(d.has_ev == 1) & ~d.buyer_type.isin([4, 6])]
    # 자기 포함: (buyer, applNo) 가 PI에 t0 전에 있는가
    m = e[['rgst_no', 'applNo', 'buyer', 'y', 'buyer_nonempty']].merge(PI.rename(columns={'name': 'buyer', 'appl_no': 'applNo'}), on=['buyer', 'applNo'], how='left')
    m = m[m.date.notna() & (m.date < f'{Y}0101')]
    selfin = m.groupby('path').rgst_no.nunique().to_dict()
    rows.append((Y, len(d), int(d.y.sum()), d.y.mean(), int(d.has_ev.sum()), len(e), selfin))
    print(Y, len(d), int(d.y.sum()), round(d.y.mean(), 6), int(d.has_ev.sum()), len(e), 'self-in-portfolio by path', selfin, flush=True)

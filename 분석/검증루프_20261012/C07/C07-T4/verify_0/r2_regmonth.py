# 반박 검증 r2: registry 월별 등록 수(2021-01~2024-12)로 잠금 위험집합 결손 추정 재확인
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
C = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\cache'
p = pd.read_parquet(C + r'\pat.parquet', columns=['rgst_no', 'rgstDate'])
m = p.rgstDate.astype(str).str[:6].value_counts().sort_index()
m = m[(m.index >= '202101') & (m.index <= '202412')]
print(m.to_string())
base = m[(m.index >= '202101') & (m.index <= '202310')].mean()
low = m[(m.index >= '202311') & (m.index <= '202412')]
print('base mean 2021-01..2023-10', round(base), 'deficit 2023-11..2024-12', round((base - low).clip(lower=0).sum()))

# 감사 3: 10차·11차 연결 로직 모의 검증(홀드아웃 자료 미사용). <=2019 10차 자료로 가짜 '11차'(2016-2019)를 만들고 ov=2016으로 연결.
# 기대: 공통 코드의 y·g1·ldev3·동료성장은 원래 패널과 동일, 11차 값을 10% 부풀려도 y·ldev3 불변.
import sys, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H); os.chdir(H)
import pandas as pd, numpy as np
import final_run as fr
k = pd.read_csv(f'{fr.D}/작업데이터/kosis_광업제조업_2008_2024.csv', dtype={'세세분류코드': str})
k = k[(k.연도 <= 2019) & (k.기준 == '10차')]
k50 = sorted(pd.read_csv(f'{fr.D}/작업데이터/kosis_광업제조업_50개산업.csv', dtype={'세세분류코드': str}).세세분류코드.unique())
fake11 = k[k.연도 >= 2016].copy(); fake11['기준'] = '11차'
scaled = '26211' if '26211' in set(k.세세분류코드) else k50[0]
fake11.loc[fake11.세세분류코드 == scaled, '값'] *= 1.1   # 차수 변경으로 수준 정의가 바뀐 경우
ren_from = "29120" if "29120" in set(k.세세분류코드) else [c for c in k.세세분류코드.unique() if c not in k50][5]
fake11.loc[fake11.세세분류코드 == ren_from, '세세분류코드'] = ren_from[:3] + '9Z'  # 11차 신설(재코드) 모사
kk = pd.concat([k[k.연도 <= 2016], fake11])
A = fr.build_panel(2019, 2019, extra=False)
B = fr.build_panel(2019, 2019, extra=False, k_all=kk, k50_codes=k50, ov=2016)
m = A.merge(B, on=['ksic', 'year'], suffixes=('_a', '_b'))
m = m[m.year.between(2017, 2019) & (m.ksic != ren_from)]
for c in ["y", "g1", "g2", "ldev3", "gemp1", "cost_ratio1"]:
    d = (m[c + '_a'] - m[c + '_b']).abs()
    bad = m.loc[d > 1e-9, 'ksic'].unique()
    print(f'{c}: 불일치 코드 {list(bad)[:5]} (n={len(bad)})')
nb = B[(B.ksic == ren_from[:3] + '9Z')][['year', 'y', 'g1']]
print('신설 코드 행:', nb[nb.year >= 2015].to_dict('records'))
print('scaled code', scaled)

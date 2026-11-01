# -*- coding: utf-8 -*-
"""출원연도 정의 차(kr_registered 출원일 vs 등록원부 applDate) 성격 확인"""
import sys, pathlib, sqlite3; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16common import *
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
pt = pd.read_sql('select rgst_no, applNo, applDate from patent', con); con.close()
kr = pd.read_parquet(KR, columns=['등록번호', '출원번호', '출원일', '출원연도', '우선권국가'])
m = kr.merge(pt, left_on='등록번호', right_on='rgst_no')
m['ry'] = pd.to_numeric(m.applDate.astype(str).str[:4], errors='coerce'); mm = m[m.ry != m['출원연도']]
print('mismatch', len(mm), 'appl_no same', float((mm['출원번호'].astype(str) == mm.applNo.astype(str)).mean()))
print('kr earlier share', float((mm['출원연도'] < mm.ry).mean()), 'median gap yrs', float((mm.ry - mm['출원연도']).median()))
print('applNo prefix (10=국내, 1020YY7=PCT 국내단계?)', mm.applNo.astype(str).str[:7].str[4:5].value_counts().head(5).to_dict())
print(mm[['출원번호', '출원일', 'applNo', 'applDate']].head(6).to_string())
out = dict(mismatch=int(len(mm)), kr_earlier_share=float((mm['출원연도'] < mm.ry).mean()), median_gap=float((mm.ry - mm['출원연도']).median()),
           applno_7th_digit=mm.applNo.astype(str).str[6:7].value_counts().head(5).to_dict())
jdump(out, CC/'d_ay.json')

# -*- coding: utf-8 -*-
"""verify_1: 부록 DATA-BIZRIGHTS(542행) 숫자 재계산 — 원천 sqlite(mode=ro) SQL + 공모전데이터/작업데이터/kr_registered.parquet(T1 캐시 미사용).
 쓰기: verify_1/v1_bizrights_out.json"""
import json, pathlib, sqlite3
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
con = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri() + '?mode=ro', uri=True)
O = {}
O['rows'] = con.execute('select count(*) from right_list').fetchone()[0]
O['prefix_rows'] = dict(con.execute('select substr(rgst_no,1,2), count(*) from right_list group by 1').fetchall())
u = pd.read_sql("select distinct rgst_no from right_list where substr(rgst_no,1,2)='10'", con).rgst_no.astype(str).str.strip()
u = u.drop_duplicates()
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['등록번호'])
k = kr['등록번호'].astype(str).str.strip().str.replace('-', '', regex=False)
O['kr_rows'] = int(len(kr)); O['kr_prefix'] = k.str[:2].value_counts().head(5).to_dict()
O['kr_sample'] = k.head(3).tolist(); O['rl_sample'] = u.head(3).tolist()
INS = set(k)
O['rl_patent_unique'] = int(len(u))
O['oos_unique'] = int((~u.isin(INS)).sum()); O['ins_unique'] = int(u.isin(INS).sum())
O['oos_share_of_unique'] = O['oos_unique'] / O['rl_patent_unique']
O['oos_share_of_rows'] = O['oos_unique'] / O['rows']
# 권리자란만
pr = pd.read_sql("select distinct rgst_no from right_list where substr(rgst_no,1,2)='10' and json_extract(raw,'$.pertDiv')='권리자란'", con).rgst_no.astype(str).str.strip().drop_duplicates()
O['rightholder_unique'] = int(len(pr)); O['rightholder_oos_unique'] = int((~pr.isin(INS)).sum())
json.dump(O, open(HERE/'v1_bizrights_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(O, ensure_ascii=False, indent=1))

# -*- coding: utf-8 -*-
"""s01: business_rights.sqlite(mode=ro)에서 디자인(30*)·실용신안(20*) 행과 원문 JSON 필드 추출 → cache/rl_dsum.parquet
   + applicant_query 표 → cache/aq.parquet"""
import sqlite3, pathlib, time
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
t0 = time.time()
con = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri() + '?mode=ro', uri=True)
pd.read_sql('select applicant_cd, applicant_name, n_in_fusion, total_count, fetched, pages, truncated, status, ts from applicant_query', con).to_parquet(HERE/'cache/aq.parquet', index=False)
q = """select applicant_cd, rgst_no, rgst_date, appl_no, appl_date, title, expire_date,
 json_extract(raw,'$.pertDiv') pert, json_extract(raw,'$.indvCorpTpcd') tp,
 json_extract(raw,'$.applicantInfo') ainfo, json_extract(raw,'$.rightHolderInfo') hinfo,
 json_extract(raw,'$.rgstStatus') st, json_extract(raw,'$.designSeq') dseq, json_extract(raw,'$.pubDate') pub,
 json_extract(raw,'$.cndrtExptnDate') exp2
 from right_list where substr(rgst_no,1,2) in ('30','20')"""
parts = []
for ch in pd.read_sql(q, con, chunksize=200000):
    parts.append(ch); print('chunk', len(ch), round(time.time() - t0), flush=True)
d = pd.concat(parts, ignore_index=True)
d['kind'] = d.rgst_no.astype(str).str[:2].map({'30': 'design', '20': 'um'})
d.to_parquet(HERE/'cache/rl_dsum.parquet', index=False)
print('rows', len(d), d.kind.value_counts().to_dict(), round(time.time() - t0))
# 전체 권리 종류 행 수(대조용)
print(pd.read_sql("select substr(rgst_no,1,2) p, count(*) n from right_list group by 1", con).to_string())

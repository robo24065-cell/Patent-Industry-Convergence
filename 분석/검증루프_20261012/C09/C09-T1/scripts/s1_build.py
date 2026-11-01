# -*- coding: utf-8 -*-
# 1단계: 분모 U(business_rights 특허, 등록<=2025-08-22, status OK)와 제공 KR 메타 캐시
import sys, os, time, json
sys.dont_write_bytecode = True; sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
t0 = time.time(); log = {}
con = duck()
kr = con.sql(f"""select appl_no, appl_year::int appl_year, nullif(rgst_no,'') rgst_no, rgst_date, pub_date, main_ipc, ksic, applicant
  from read_parquet('{META}') where cc='KR'""").df()
kr.to_parquet(os.path.join(CACHE, "kr_meta.parquet"), index=False)
log["kr_meta_rows"] = len(kr); log["kr_meta_distinct_appl"] = int(kr.appl_no.nunique())
print("kr meta", len(kr), round(time.time() - t0), flush=True)
c = ro(BR)
log["applicant_query_status"] = dict(c.execute("select status, count(*) from applicant_query group by 1").fetchall())
log["right_list_rows_all"] = c.execute("select count(*) from right_list").fetchone()[0]
q = f"""select r.applicant_cd, r.rgst_no, r.rgst_date, r.appl_no, r.appl_date, r.title,
 json_extract(r.raw,'$.pertDiv') pert, json_extract(r.raw,'$.applicantInfo') ainfo, json_extract(r.raw,'$.rightHolderInfo') rinfo,
 json_extract(r.raw,'$.indvCorpTpcd') ctype, json_extract(r.raw,'$.rgstStatus') rstat
 from right_list r join applicant_query q on q.applicant_cd=r.applicant_cd
 where q.status='OK' and substr(r.rgst_no,1,2)='10' and r.rgst_date<='{DATA_END}'"""
w = None; n = 0
for ch in pd.read_sql(q, c, chunksize=400000):
    t = pa.Table.from_pandas(ch, preserve_index=False)
    if w is None: w = pq.ParquetWriter(os.path.join(CACHE, "U_raw.parquet"), t.schema)
    w.write_table(t); n += len(ch); print("U chunk", n, round(time.time() - t0), flush=True)
w.close()
log["U_rows"] = n
log["right_list_patent_rows_all_dates"] = c.execute("select count(*) from right_list where substr(rgst_no,1,2)='10'").fetchone()[0]
log["right_list_patent_rows_after_end"] = c.execute(f"select count(*) from right_list where substr(rgst_no,1,2)='10' and rgst_date>'{DATA_END}'").fetchone()[0]
rf = ro(RF)
pd.read_sql("select rgst_no, applNo, applDate, ipcCd, ksic, rgstDate from patent", rf).to_parquet(os.path.join(CACHE, "rf_patent.parquet"), index=False)
pd.read_sql("select rgst_no, seq, applicantCd, applicantName, applicantNatl from applicant", rf).to_parquet(os.path.join(CACHE, "rf_applicant.parquet"), index=False)
log["sec"] = round(time.time() - t0)
json.dump(log, open(os.path.join(CACHE, "s1_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(log)

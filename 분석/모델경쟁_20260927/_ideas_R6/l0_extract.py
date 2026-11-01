# -*- coding: utf-8 -*-
"""메타 → 국가별 (cc, appl_no) 1행 캐시. 🔒 첫 공개일 vis = min(pub_date, rgst_date) < 2023-01-01 인 행만 저장(봉인).
사용: python l0_extract.py KR   (KR/US/JP/CN/EP 각각 1회)"""
import sys, os, time, duckdb
from r6_common import META, CACHE, SEAL_DATE

cc = sys.argv[1]
t = time.time()
con = duckdb.connect()
con.sql("set threads=2; set memory_limit='3GB'; set preserve_insertion_order=false")
con.sql(f"set temp_directory='{CACHE}/_tmp'".replace("\\", "/"))
out = os.path.join(CACHE, f"p_{cc}.parquet").replace("\\", "/")
con.sql(f"""
copy (
  select appl_no, min(ksic) ksic, count(distinct ksic)::int nk,
         any_value(appl_date) ad, any_value(appl_year)::int ay,
         any_value(pub_date) pubd, any_value(rgst_date) rgd,
         any_value(prio_cc) pcc, any_value(prio_date) pdt,
         any_value(main_ipc) ipc, any_value(ipc_all) ipcs, any_value(applicant) app, any_value(family) fam
  from read_parquet('{META}')
  where cc='{cc}'
  group by appl_no
  having least(coalesce(nullif(trim(any_value(pub_date)),''),'99999999'), coalesce(nullif(trim(any_value(rgst_date)),''),'99999999')) < '{SEAL_DATE}'
) to '{out}' (format parquet)
""")
n = con.sql(f"select count(*), max(least(coalesce(nullif(trim(pubd),''),'99999999'), coalesce(nullif(trim(rgd),''),'99999999'))) from '{out}'").fetchone()
assert n[1] < SEAL_DATE, n
print(cc, n, round(time.time() - t, 1), "s")

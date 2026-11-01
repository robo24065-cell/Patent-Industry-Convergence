# -*- coding: utf-8 -*-
# 1단계: 첫 출원인 이름 → N0·N2 id 사전, KSIC×cc×연도×출원인 집계 (읽기 전용 원자료; 쓰기 cache)
import sys, os, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import duckdb, pandas as pd, numpy as np
t = time.time()
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false")
FIRST = "nullif(trim(split_part(coalesce(applicant,''),'|',1)),'')"
nm = con.sql(f"select cc, {FIRST} raw, count(*) n from read_parquet('{META}') group by 1,2").df()
print("names", len(nm), round(time.time() - t), flush=True)
nm["n2"] = [norm_n2(x) if isinstance(x, str) else None for x in nm.raw.values]
nm["id_n0"] = np.where(nm.raw.isna(), -1, nm.groupby(["cc", "raw"], dropna=False).ngroup())
k2 = nm.cc + "\x01" + nm.n2.fillna("")
nm["id_n2"] = np.where(nm.n2.isna(), -1, pd.factorize(k2)[0])
nm.to_parquet(os.path.join(CACHE, "s1_names.parquet"), index=False)
print("norm", round(time.time() - t), nm.groupby("cc").agg(raw=("raw", "nunique"), n2=("n2", "nunique")).to_string(), flush=True)
con.register("nm", nm[["cc", "raw", "id_n0", "id_n2"]])
inv = """case when m.cc='KR' then substr(m.appl_no,1,2)='10'
 when m.cc='CN' then length(m.appl_no) in (12,14) and substr(m.appl_no,5,1) in ('1','8')
 when m.cc='JP' then length(m.appl_no) in (8,17) else true end"""
q = f"""
copy (
select m.ksic, m.cc, m.appl_year yr, coalesce(nm.id_n2,-1) a2, coalesce(nm.id_n0,-1) a0,
  ({inv}) is_inv, (coalesce(nullif(split_part(m.prio_cc,'|',1),''), m.cc)='KR') kro,
  count(*) n_rows, count(distinct m.appl_no) n_uniq
from read_parquet('{META}') m left join nm on m.cc=nm.cc and {FIRST.replace('applicant','m.applicant')} is not distinct from nm.raw
where m.appl_year between 1995 and 2025
group by all) to '{os.path.join(CACHE, 's1_agg.parquet').replace(chr(92), '/')}' (format parquet)"""
con.sql(q)
a = pd.read_parquet(os.path.join(CACHE, "s1_agg.parquet"))
print("agg rows", len(a), "rows sum", a.n_rows.sum(), "uniq sum", a.n_uniq.sum(), round(time.time() - t), flush=True)

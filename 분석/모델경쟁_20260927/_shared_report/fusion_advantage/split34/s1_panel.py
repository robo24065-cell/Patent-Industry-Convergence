# -*- coding: utf-8 -*-
r"""split34 1단계: 제공 융합데이터 → KSIC 5자리 × 국가 × 출원연도 패널.
- 키: (cc, appl_no)를 KSIC 안에서 고유화. 출원연도 기준.
- 한국 기원(origin) = 우선권 첫 국가코드, 없으면 출원국(국내 최초출원으로 간주).
- 연계표 그룹(공식 IPC-KSIC 2013, label_validity의 lv_common.GROUP_PREFIX)도 붙인다.
쓰기는 이 폴더에만. 원본 parquet는 읽기만."""
import os, sys, time, json
sys.dont_write_bytecode = True
import duckdb, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group  # 읽기만(import)
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
t = time.time()
con = duckdb.connect()
con.sql("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
q = f"""
with u as (
  select distinct ksic, cc, appl_no, appl_year,
     coalesce(nullif(split_part(prio_cc,'|',1),''), cc) as origin
  from read_parquet('{META}') where appl_year between 1995 and 2024
)
select ksic, cc, appl_year as year, count(*) n, sum(case when origin='KR' then 1 else 0 end) n_kro
from u group by 1,2,3
"""
P = con.sql(q).df()
P["grp"] = P.ksic.map(ksic_to_group)
P.to_parquet(os.path.join(HERE, "panel_ksic_cc_year.parquet"), index=False)
print(P.shape, P.ksic.nunique(), P.grp.nunique(), round(time.time() - t), "s")
print(P.groupby("cc").n.sum())

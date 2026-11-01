import os
CACHE=os.environ.get("T3_CACHE","cache"); MAXYEAR=int(os.environ.get("T3_MAXYEAR","2018")); T0MAX=int(os.environ.get("T3_T0MAX","2015"))
os.makedirs(CACHE,exist_ok=True)
# 기본 행 캐시: (ksic, mg, cc, year, appl_no, app1, prio_for, ipc_all, title) — 출원연도 2000~2018만 (🔒 2019~ 결과변수 미접근)
import duckdb, time
t=time.time()
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
M="../_shared/meta/*.parquet"
con.execute(f"""
copy (
 select ksic, cc, appl_year::int as year, appl_no,
   regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g') as ipc_raw,
   trim(split_part(applicant,'|',1)) as app1,
   case when prio_cc is null then 0
        when regexp_replace(replace(prio_cc,cc,''),'\|','','g')<>'' then 1 else 0 end as prio_for,
   ipc_all, title
 from read_parquet('{M}')
 where appl_year between 2000 and {MAXYEAR}
) to '{CACHE}/base_rows.parquet' (format parquet, compression zstd)""")
print('base done', time.time()-t)
con.execute(f"""create view b as select *, split_part(ipc_raw,'/',1) as mg from read_parquet('{CACHE}/base_rows.parquet')""")
print(con.sql("select count(*), count(distinct ksic||mg), count(*) filter (where ipc_raw is null or ipc_raw='') from b").df())
print(con.sql("select mg, count(*) n from b group by 1 order by n desc limit 10").df())
print(time.time()-t)

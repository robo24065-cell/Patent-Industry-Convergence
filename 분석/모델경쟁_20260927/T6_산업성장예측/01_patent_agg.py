# 5개국 메타 → (ksic, cc, appl_year) 고유 출원 수, 해외우선권 비중, 출원인 수 집계. 결과: cache/pat_ksic_cc_year.parquet
import duckdb, os, time
t=time.time()
B=r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927"
os.makedirs('cache',exist_ok=True)
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
q=f"""
copy (
select ksic, cc, appl_year::INT as year,
  count(distinct appl_no) as n_app,
  count(distinct case when prio_cc is not null and prio_cc<>'' and split_part(prio_cc,'|',1)<>cc then appl_no end) as n_foreignprio,
  count(distinct case when rgst_no is not null and rgst_no<>'' then appl_no end) as n_rgst,
  count(distinct applicant) as n_applicant,
  count(distinct family) as n_family
from read_parquet('{B}/_shared/meta/*.parquet')
where appl_year between 2000 and 2022
group by 1,2,3) to 'cache/pat_ksic_cc_year.parquet' (format parquet)
"""
con.sql(q)
print(con.sql("select count(*), count(distinct ksic), min(year), max(year) from 'cache/pat_ksic_cc_year.parquet'").fetchall())
print('sec',time.time()-t)

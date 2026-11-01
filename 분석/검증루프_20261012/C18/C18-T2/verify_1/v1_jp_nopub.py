import duckdb,sys
sys.stdout.reconfigure(encoding='utf-8')
M='C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
c=duckdb.connect(); c.execute("set threads=2; set memory_limit='3GB'")
print(c.sql(f"""select substr(rgst_date,1,6) ym, (pub_date is null or pub_date='') nopub, appl_no like '%U' u, count(*), min(appl_year), median(appl_year), max(appl_year)
 from read_parquet('{M}') where cc='JP' and substr(rgst_date,1,6) in ('202010','202012','202101','202103','202105','202106','202108','202203','202205','202208')
 group by all order by all""").fetchall())

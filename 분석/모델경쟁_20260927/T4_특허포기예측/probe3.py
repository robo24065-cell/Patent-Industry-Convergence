import duckdb
M=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta\*.parquet"
duckdb.sql("SET threads=2; SET memory_limit='3GB'")
print(duckdb.sql(f"select cc,appl_no,family,prio_cc,prio_date,rgst_no,ksic from read_parquet('{M}') where cc='KR' limit 8").df().to_string())
print(duckdb.sql(f"select cc,appl_no,family,prio_cc from read_parquet('{M}') where cc='US' limit 5").df().to_string())
print(duckdb.sql(f"select cc, count(*), count(family), avg(case when family is null or family='' then 0 else 1 end) from read_parquet('{M}') group by cc").df())

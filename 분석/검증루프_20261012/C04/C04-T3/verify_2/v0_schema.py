import duckdb, sys
sys.stdout.reconfigure(encoding='utf-8')
c = duckdb.connect(); c.execute("set memory_limit='3GB'; set threads=2")
p = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T3_기술셀부상예측/R3/cache_pub_final/base_rows.parquet'
print(c.sql(f"describe select * from read_parquet('{p}')").df().to_string())
print(c.sql(f"select cc, min(year), max(year), count(*) from read_parquet('{p}') group by 1 order by 1").df())
print(c.sql(f"select * from read_parquet('{p}') where cc='KR' limit 5").df().to_string())

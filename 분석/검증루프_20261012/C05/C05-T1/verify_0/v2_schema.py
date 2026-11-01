import duckdb, sys
sys.stdout.reconfigure(encoding='utf-8')
BR = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet".replace('\\', '/')
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
print(con.sql(f"describe select * from read_parquet('{BR}')"))
print(con.sql(f"select * from read_parquet('{BR}') limit 5"))
print(con.sql(f"select min(year), max(year), count(*) from read_parquet('{BR}')"))
print(con.sql(f"select count(*) from (select cc, appl_no, count(distinct year) ny from read_parquet('{BR}') group by all) where ny>1"))

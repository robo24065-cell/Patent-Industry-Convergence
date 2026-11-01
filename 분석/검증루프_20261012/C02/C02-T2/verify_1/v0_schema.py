import duckdb, sys
sys.stdout.reconfigure(encoding='utf-8')
META = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta\*.parquet".replace("\\","/")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
print(con.sql(f"describe select * from read_parquet('{META}')").df().to_string())
print(con.sql(f"select * from read_parquet('{META}') where cc='CN' limit 3").df().T.to_string())

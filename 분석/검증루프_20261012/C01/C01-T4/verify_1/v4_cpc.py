import duckdb, os
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
print(con.sql(f"select count(*), sum((main_cpc is not null and main_cpc<>'')::int), sum((title is not null)::int), sum((family is not null and family<>'')::int) from read_parquet('{META}') where cc='CN' and main_ipc is null").fetchall())
print(con.sql(f"select main_cpc, pub_date, rgst_date from read_parquet('{META}') where cc='CN' and main_ipc is null limit 3").fetchall())

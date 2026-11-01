import duckdb, os
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
print(con.sql(f"describe select * from read_parquet('{META}')").fetchall())
# IPC format patterns per country
print(con.sql(f"""select cc, regexp_replace(regexp_replace(coalesce(main_ipc,'<NULL>'),'[0-9]','9','g'),'[A-Z]','A','g') pat, count(*) n
 from read_parquet('{META}') group by 1,2 qualify row_number() over (partition by cc order by n desc)<=6 order by cc, n desc""").fetchall())

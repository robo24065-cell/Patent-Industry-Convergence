import duckdb, os
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
print(con.sql(f"""select cc, year(least(try_strptime(nullif(pub_date,''),'%Y%m%d'), try_strptime(nullif(rgst_date,''),'%Y%m%d'))) y,
  count(*) n, sum(regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z][0-9]+/')::int) nosp, sum(regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z] *0[0-9]+/')::int) lead0,
  sum((main_ipc is null)::int) nnull
  from read_parquet('{META}') group by 1,2 having nosp>0 or lead0>0 or nnull>0 order by 1,2""").fetchall())
print(con.sql(f"select cc, main_ipc, pub_date from read_parquet('{META}') where regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z][0-9]+/') limit 5").fetchall())
print(con.sql(f"select cc, main_ipc, pub_date from read_parquet('{META}') where cc='US' and regexp_matches(main_ipc,'^[A-H][0-9]{{2}}[A-Z] *0[0-9]+/') limit 5").fetchall())

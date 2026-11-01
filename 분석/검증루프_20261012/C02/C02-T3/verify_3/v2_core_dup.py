import sys, json, duckdb
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
META = (B + r"\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
q = lambda s: con.sql(s).fetchall()
R = {}
# CN: 12-char core (strip check digit) duplicates
R["cn_core_dup"] = q(f"""select count(*), sum((nk>1)::int) from (select substr(appl_no,1,12) core, count(*) n, count(distinct ksic) nk from read_parquet('{META}') where cc='CN' and length(appl_no) in (12,14) group by 1 having count(*)>1)""")
# per cc: strip everything after '.' and non-digits
R["core_dup_all"] = q(f"""select cc, count(*), sum((nk>1)::int) from (select cc, regexp_replace(split_part(appl_no,'.',1),'[^0-9A-Za-z]','','g') core, count(*) n, count(distinct ksic) nk from read_parquet('{META}') group by all having count(*)>1) group by 1""")
R["files"] = q(f"select count(distinct filename) from read_parquet('{META}', filename=true)")
R["ksic_per_file"] = q(f"select max(nk) from (select filename, count(distinct ksic) nk from read_parquet('{META}', filename=true) group by 1)")
json.dump(R, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in R.items(): print(k, v)

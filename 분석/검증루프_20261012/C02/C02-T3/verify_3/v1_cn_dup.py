import sys, json, duckdb
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
META = (B + r"\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
R = {}
q = lambda s: con.sql(s).fetchall()
R["total_rows"] = q(f"select count(*) from read_parquet('{META}')")
R["cn_len_d5"] = q(f"select length(appl_no) l, substr(appl_no,5,1) d5, count(*), sum((pub_date is null or pub_date='')::int) nopub, min(appl_no), max(appl_no) from read_parquet('{META}') where cc='CN' group by all order by 3 desc limit 30")
# utility models by appl_year: pub presence
R["cn_um_by_ay"] = q(f"""select appl_year, count(*), sum((pub_date is not null and pub_date<>'')::int) haspub, sum((rgst_date is not null and rgst_date<>'')::int) hasrg
  from read_parquet('{META}') where cc='CN' and length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9') group by 1 order by 1""")
R["cn_um_pubyear"] = q(f"""select substr(pub_date,1,4), count(*) from read_parquet('{META}') where cc='CN' and length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9') and pub_date<>'' and pub_date is not null group by 1 order by 1""")
R["cn_inv_missing_by_ay"] = q(f"""select appl_year, count(*), round(avg((pub_date is null or pub_date='')::int),4) from read_parquet('{META}')
  where cc='CN' and length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') and appl_year between 2008 and 2018 group by 1 order by 1""")
# duplicates
R["dup_cc_appl_ksic2"] = q(f"select count(*) from (select cc, appl_no from read_parquet('{META}') group by all having count(distinct ksic)>1)")
R["dup_cc_appl_rows2"] = q(f"select count(*), sum(n) from (select cc, appl_no, count(*) n from read_parquet('{META}') group by all having count(*)>1)")
R["distinct_cc_appl"] = q(f"select count(*) from (select distinct cc, appl_no from read_parquet('{META}'))")
# normalized appl_no (strip non-alnum, upper) dup check
R["dup_norm_ksic2"] = q(f"select count(*) from (select cc, upper(regexp_replace(appl_no,'[^0-9A-Za-z]','','g')) a from read_parquet('{META}') group by all having count(distinct ksic)>1)")
# CN without check digit vs with: 13-char forms 'xxxxxxxxxxxx.x'? count by length overall
R["cn_lengths"] = q(f"select length(appl_no), count(*) from read_parquet('{META}') where cc='CN' group by 1 order by 2 desc")
# same family / title across ksic? e.g. same cc+title+appl_date in multiple ksic
R["dup_title_date_ksic2"] = q(f"select cc, count(*) from (select cc, title, appl_date from read_parquet('{META}') where title is not null and title<>'' group by all having count(distinct ksic)>1 and count(distinct appl_no)>1) group by 1")
json.dump(R, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in R.items(): print(k, v)

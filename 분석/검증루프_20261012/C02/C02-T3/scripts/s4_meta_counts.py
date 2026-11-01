# 1장 데이터 숫자 원천 재계산: 행 수·국가 비중, CN 문서종류, KR 등록, 라벨 수, 등록번호 패밀리 일치율
import sys, json, time, duckdb
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = B + r"\모델경쟁_20260927"; OUT = B + r"\검증루프_20261012\C02\C02-T3"
META = (MC + r"\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false; SET preserve_insertion_order=false")
t = time.time(); R = {}
R["rows_by_cc"] = dict(con.sql(f"select cc, count(*) from read_parquet('{META}') group by cc").fetchall())
R["rows_total"] = sum(R["rows_by_cc"].values())
R["n_ksic"] = con.sql(f"select count(distinct ksic) from read_parquet('{META}')").fetchone()[0]
R["cn_doctype"] = dict(con.sql(f"""select case when length(appl_no) in (12,14) then substr(appl_no,5,1) else 'len'||length(appl_no)::varchar end k, count(*)
   from read_parquet('{META}') where cc='CN' group by 1 order by 2 desc""").fetchall())
R["kr_doctype"] = dict(con.sql(f"select substr(appl_no,1,2), count(*) from read_parquet('{META}') where cc='KR' group by 1").fetchall())
R["kr_rgst_unique"] = con.sql(f"select count(distinct rgst_no) from read_parquet('{META}') where cc='KR' and rgst_no is not null and rgst_no<>''").fetchone()[0]
R["kr_appl_unique"] = con.sql(f"select count(distinct appl_no) from read_parquet('{META}') where cc='KR'").fetchone()[0]
R["kr_labels_per_appl"] = con.sql(f"select avg(k), max(k) from (select appl_no, count(distinct ksic) k from read_parquet('{META}') where cc='KR' group by 1)").fetchone()
R["dup_cc_appl_ksic"] = con.sql(f"select count(*) from (select cc, appl_no, ksic, count(*) c from read_parquet('{META}') group by all having count(*)>1)").fetchone()[0]
R["multi_ksic_docs_by_cc"] = dict(con.sql(f"select cc, avg((k>1)::int) from (select cc, appl_no, count(distinct ksic) k from read_parquet('{META}') group by 1,2) group by 1").fetchall())
print(R, round(time.time() - t), flush=True)
# 등록번호 기반 패밀리 재연결(lo_fam 정의를 독립 구현): 해외 family 토큰 KR(10|20)+7자리 ↔ KR rgst_no 앞 9자리
con.sql(f"create temp table kr as select appl_no kr_appl_no, substr(rgst_no,1,9) r9, ksic kk from read_parquet('{META}') where cc='KR' and rgst_no is not null and length(rgst_no)>=9")
con.sql(f"""create temp table tok as select distinct cc, appl_no, ksic fk, substr(t,3) r9 from (select cc, appl_no, ksic, unnest(string_split(family,'|')) t
   from read_parquet('{META}') where cc<>'KR' and family like '%KR%') where regexp_full_match(t,'KR(10|20)[0-9]{{7}}')""")
R["fam_links"] = [list(r) for r in con.sql("select cc, count(*), avg((fk=kk)::int) from tok join kr using (r9) group by cc order by cc").fetchall()]
json.dump(R, open(OUT + r"\s4_meta_counts_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(R["fam_links"], round(time.time() - t))

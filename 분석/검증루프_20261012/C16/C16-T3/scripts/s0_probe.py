# C16-T3 s0: 제공 메타 CN·KR 형식 탐색(판정 없음). 사용: python -I s0_probe.py <meta_glob> <out_json>
import sys, json, duckdb
M, OUT = sys.argv[1].replace('\\', '/'), sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
r = {}
r['cn_len_type'] = con.execute(f"""select length(appl_no) L, substr(appl_no, 7, 1) t6, count(*) n, min(appl_no), max(appl_no)
  from read_parquet('{M}') where cc='CN' group by 1,2 order by n desc limit 20""").fetchall()
r['cn_sample'] = con.execute(f"""select appl_no, appl_year, pub_date, rgst_date, prio_cc, applicant from read_parquet('{M}')
  where cc='CN' and pub_date between '20190101' and '20191231' using sample 15 rows""").fetchall()
r['kr_len_pref'] = con.execute(f"""select length(appl_no) L, substr(appl_no,1,2) p, count(*) n, count(distinct appl_no) nu, min(appl_no), max(appl_no)
  from read_parquet('{M}') where cc='KR' group by 1,2 order by n desc""").fetchall()
r['cn_prio_top'] = con.execute(f"""select prio_cc, count(*) from read_parquet('{M}') where cc='CN' and pub_date>='20150101' group by 1 order by 2 desc limit 15""").fetchall()
r['cn_sep'] = con.execute(f"""select sum((applicant like '%|%')::int), sum((applicant like '%;%')::int), sum((applicant is null)::int), count(*)
  from read_parquet('{M}') where cc='CN' and pub_date>='20150101'""").fetchall()
json.dump(r, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(r, ensure_ascii=False, indent=1, default=str))

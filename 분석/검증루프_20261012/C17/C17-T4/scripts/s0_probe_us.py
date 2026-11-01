# C17-T4 s0: 제공 US 메타 구조 탐색 (읽기 전용)
import duckdb, json, sys
META = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta\*.parquet"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C17\C17-T4\cache\s0_probe_us.json"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=2")
res = {}
q = lambda s: con.execute(s).fetchall()
base = f"(select * from read_parquet('{META}') where cc='US')"
res['n_rows'] = q(f"select count(*), count(distinct appl_no) from {base}")
res['rgst_nonnull'] = q(f"select count(*) filter (where rgst_no is not null and rgst_no<>''), count(*) filter (where rgst_date is not null and rgst_date<>'') from {base}")
res['sample'] = q(f"select appl_no, appl_date, rgst_no, rgst_date, pub_date, applicant, main_ipc, main_cpc, family, prio_cc, ksic from {base} using sample 15 rows")
res['appl_no_len'] = q(f"select length(appl_no) l, count(*) from {base} group by l order by 2 desc limit 10")
res['rgst_no_len'] = q(f"select length(rgst_no) l, count(*) from {base} group by l order by 2 desc limit 10")
res['rgst_year'] = q(f"select substr(rgst_date,1,4) y, count(*) from {base} group by y order by y")
res['appl_year_reg'] = q(f"select appl_year, count(*), count(*) filter (where rgst_no is not null and rgst_no<>'') from {base} group by appl_year order by appl_year")
res['pub_year'] = q(f"select substr(pub_date,1,4) y, count(*) from {base} group by y order by y")
res['dup_appl_ksic'] = q(f"select count(*) from (select appl_no, count(distinct ksic) c from {base} group by appl_no having c>1)")
res['rgst_prefix'] = q(f"select substr(rgst_no,1,2) p, count(*) from {base} where rgst_no is not null and rgst_no<>'' group by p order by 2 desc limit 15")
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str)[:6000])

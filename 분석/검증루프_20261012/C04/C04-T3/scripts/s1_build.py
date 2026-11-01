# C04-T3 s1: 동결 캐시(base_rows, 공개연도 ≤2024)에서 셀 패널과 KR 문서표(출원인 포함)를 만든다. 읽기 전용 입력, 쓰기는 C04-T3/cache 만.
# 셀 정의는 verify_final_r3d.py / C02 s8a 와 같다: mgn(ipc_raw), (ksic,mg,cc,year,appl_no) 고유.
import os, sys, time, duckdb
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3"); C = os.path.join(W, 'cache'); os.makedirs(C, exist_ok=True)
CA = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub_final\base_rows.parquet").replace("\\", "/")
META = os.path.join(ROOT, r"모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
Cs = C.replace("\\", "/")
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
# 1) 셀×국가×연도 패널
con.execute(f"""copy (select ksic, mg, cc, year, count(distinct appl_no) n from (select ksic, mgn(ipc_raw) mg, cc, year, appl_no
   from read_parquet('{CA}')) where mg is not null group by all) to '{Cs}/panel.parquet' (format parquet)""")
# 2) KR 문서표: 셀·공개연도·출원번호·출원인(원문)
con.execute(f"""copy (select ksic, mg, year, appl_no, any_value(app1) app_raw, count(distinct app1) n_app_vals from (
   select ksic, mgn(ipc_raw) mg, year, appl_no, app1 from read_parquet('{CA}') where cc='KR') where mg is not null group by ksic, mg, year, appl_no)
   to '{Cs}/kr_docs.parquet' (format parquet)""")
# 3) CN 공개연도 2021 IPC 결측 (ksic 단위) — base_rows 는 ipc_raw 공란 행을 포함한다
con.execute(f"""copy (select ksic, year, count(distinct case when ipc_raw is null or ipc_raw='' then appl_no end) n_null,
   count(distinct case when mgn(ipc_raw) is not null then appl_no end) n_valid, count(distinct appl_no) n_all
   from read_parquet('{CA}') where cc='CN' and year between 2017 and 2024 group by all) to '{Cs}/cn_missing_ky.parquet' (format parquet)""")
print(con.sql(f"select cc, count(*), sum(n) from read_parquet('{Cs}/panel.parquet') group by 1 order by 1").df())
print(con.sql(f"select count(*) n, sum((n_app_vals>1)::int) multi, sum((app_raw is null or trim(app_raw)='')::int) miss, min(year), max(year) from read_parquet('{Cs}/kr_docs.parquet')").df())
print(con.sql(f"select year, sum(n_null)*1.0/sum(n_all) miss from read_parquet('{Cs}/cn_missing_ky.parquet') group by 1 order by 1").df())
# 4) meta 직접(대조): KR 고유 출원·applicant 구분자 확인
print(con.sql(f"""select count(*) n, count(distinct appl_no) nu, sum((applicant like '%|%')::int) pipe, sum((applicant is null or trim(applicant)='')::int) miss,
   sum((applicant like '%&#%' or applicant like '%&quot;%' or applicant like '%&amp;%' or applicant like '%&lt;%')::int) html
   from read_parquet('{META}') where cc='KR'""").df())
print('sec', round(time.time() - t, 1))

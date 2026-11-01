# C02-T2 s8a: T3 셀 패널 재구성(읽기 전용, 모형·재학습 없음) — build_base_pub.py + build_panel.py 의 n(셀×국가×공개연도 고유 출원) 정의 그대로
# 추가: CN 결측(ipc_raw 공란) 고유 출원 수 (ksic × 공개연도 × 문서종류), CN 비결측 셀 수 (문서종류별)
import os, sys, time
import duckdb
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2"); C = os.path.join(W, 'cache').replace("\\", "/")
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
con.execute(rf"""create temp table b as
 select ksic, cc, py::int as year, appl_no,
   regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g') as ipc_raw,
   case when cc='CN' and length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9') then 'um' else 'inv' end dt
 from (select *, left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4) as py
       from read_parquet('{META}') where appl_year >= 2000)
 where py is not null and py::int between 2000 and 2024""")
con.execute("create temp table b2 as select *, mgn(ipc_raw) mg from b")
# 셀 패널 (build_panel: mgn(ipc_raw) not null, (ksic,mg,cc,year,appl_no) dedupe 후 count(*))
con.execute(f"""copy (select ksic, mg, cc, year, count(*) n from (select distinct ksic, mg, cc, year, appl_no from b2 where mg is not null) group by all)
               to '{C}/s8_panel.parquet' (format parquet)""")
# CN 문서종류별 셀 패널
con.execute(f"""copy (select ksic, mg, year, dt, count(*) n from (select distinct ksic, mg, year, appl_no, dt from b2 where cc='CN' and mg is not null) group by all)
               to '{C}/s8_cn_panel_dt.parquet' (format parquet)""")
# CN 결측(ipc_raw 공란 또는 '' ) / 파싱불가 고유 출원 (ksic×year×dt)
con.execute(f"""copy (select ksic, year, dt, count(distinct case when ipc_raw is null or ipc_raw='' then appl_no end) n_null,
                       count(distinct case when not (ipc_raw is null or ipc_raw='') and mg is null then appl_no end) n_unparse,
                       count(distinct case when mg is not null then appl_no end) n_valid
               from b2 where cc='CN' group by all) to '{C}/s8_cn_missing_ky.parquet' (format parquet)""")
print(con.sql(f"select cc, count(*), sum(n) from read_parquet('{C}/s8_panel.parquet') group by 1 order by 1").df())
print(con.sql(f"select year, sum(n_null), sum(n_unparse), sum(n_valid) from read_parquet('{C}/s8_cn_missing_ky.parquet') where year>=2018 group by 1 order by 1").df())
print(round(time.time() - t, 1))

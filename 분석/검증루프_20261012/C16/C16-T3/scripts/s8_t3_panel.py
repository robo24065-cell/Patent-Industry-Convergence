"""s8: T3 셀×국가×공개연도 출원인 집계 + CN 출원인1 조직/개인 분리 (build_panel.py 규칙 사본, C09-T3 s1_panel 방식).
사용: python -I s8_t3_panel.py dev|fin → cache/t3panel_{dev|fin}.parquet, cache/t3_cn_orgshare_{which}.csv
열: n, n_app, n_newapp_apps, n_newapp (S0, 공식 규칙 재현)
    CN 전용: n_org, n_ind, n_miss, nna_org, nna_ind (새 출원인 문서 수), nn_org, nn_ind (새 출원인 고유 수)
입력: 트랙 R3/cache_pub(_final)/base_rows.parquet (읽기 전용)
"""
import sys, os, time, duckdb, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
which = sys.argv[1]
src = CP_DEV if which == 'dev' else CP_FIN
B = ddb(os.path.join(src, 'base_rows.parquet'))
OUT = ddb(os.path.join(CACHE, f't3panel_{which}.parquet'))
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(f"set temp_directory='{ddb(os.path.join(CACHE, 'duck_tmp'))}'")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
con.execute(f"""create temp table a as select ksic, mgn(ipc_raw) mg, cc, year, appl_no, any_value(app1) app1
  from read_parquet('{B}') where mgn(ipc_raw) is not null group by all""")
nm = con.execute("select distinct app1 from a where cc='CN' and app1 is not null and app1<>''").fetchdf()
nm['org'] = nm.app1.map(is_org_cn)
con.register('nmo', nm)
print('a', con.sql("select count(*) from a").fetchone(), 'CN names', len(nm), 'org share', round(nm.org.mean(), 4), round(time.time() - t), flush=True)
con.execute("""create temp table a2 as select a.*, (app1 is null or app1='') miss,
   coalesce(case when cc='CN' then nmo.org end, false) org from a left join nmo using(app1)""")
con.execute("create temp table fa as select ksic, mg, cc, app1, min(year) fy from a2 where not miss group by all")
con.execute("""create temp table p as
 select a2.ksic, a2.mg, a2.cc, a2.year, count(*) n, count(distinct a2.app1) n_app,
  sum(case when fa.fy=a2.year then 1 else 0 end) n_newapp_apps,
  count(distinct case when fa.fy=a2.year then a2.app1 end) n_newapp,
  sum((not miss and org)::int) n_org, sum((not miss and not org)::int) n_ind, sum(miss::int) n_miss,
  sum(case when fa.fy=a2.year and org then 1 else 0 end) nna_org,
  sum(case when fa.fy=a2.year and not org then 1 else 0 end) nna_ind,
  count(distinct case when fa.fy=a2.year and org then a2.app1 end) nn_org,
  count(distinct case when fa.fy=a2.year and not org then a2.app1 end) nn_ind
 from a2 left join fa using(ksic,mg,cc,app1) group by all""")
con.execute(f"copy p to '{OUT}' (format parquet)")
# CN 연도별 조직 비중(패널 문서 기준, 결측 제외)
con.execute(f"""copy (select year, sum(n_org) n_org, sum(n_ind) n_ind, sum(n_miss) n_miss, sum(n_org)*1.0/nullif(sum(n_org)+sum(n_ind),0) org_share
   from read_parquet('{OUT}') where cc='CN' group by year order by year) to '{ddb(os.path.join(CACHE, f't3_cn_orgshare_{which}.csv'))}' (header)""")
print(con.sql(f"select year, org_share, n_org+n_ind+n_miss n from read_csv_auto('{ddb(os.path.join(CACHE, f't3_cn_orgshare_{which}.csv'))}')").df().to_string())
print('done', round(time.time() - t), flush=True)

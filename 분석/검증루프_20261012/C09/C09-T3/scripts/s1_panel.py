"""s1: 시나리오용 셀×국가×공개연도 출원인 집계 (build_panel.py 규칙 사본 + 시나리오 열).
사용: python s1_panel.py dev|fin   → cache/panel_scen_{dev|fin}.parquet
입력: 트랙 R3/cache_pub(_final)/base_rows.parquet (읽기 전용)
열:
  n                       문서 수(출원 단위 dedupe, build_panel 과 같음)
  n_null                  출원인 결측 문서
  n_usold                 US 출원연도 ≤2011 문서 (S1 제외 대상)
  n_cnmiss59              CN 출원연도 2005–09 출원인 결측 문서 (S2 분모 제외)
  n_corp                  법인형 이름 문서(common.CORP_RE)
  n_app, n_newapp_apps, n_newapp       S0 (공식 규칙 재현)
  n_newapp_apps_s1, n_newapp_s1        S1: US 출원 ≤2011 문서를 출원인 이력에서 제거하고 첫 등장 재계산
  n_newapp_apps_corp                   법인형 이름 문서만으로 첫 등장 재계산(데이터 확인용)
"""
import sys, os, time, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
which = sys.argv[1]
src = CP_DEV if which == 'dev' else CP_FIN
B = ddb(os.path.join(src, 'base_rows.parquet'))
OUT = ddb(os.path.join(CACHE, f'panel_scen_{which}.parquet'))
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(f"set temp_directory='{ddb(os.path.join(CACHE, 'duck_tmp'))}'")
# build_panel.py 의 mgn 매크로 그대로
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
corp_case = "case " + " ".join(f"when cc='{c}' then regexp_matches(app1, '{rg.replace(chr(39), chr(39)*2)}')" for c, rg in CORP_RE.items()) + " else false end"
con.execute(f"""create temp table a as select ksic, mgn(ipc_raw) mg, cc, year, appl_no, any_value(app1) app1, min(appl_year) appl_year
  from read_parquet('{B}') where mgn(ipc_raw) is not null group by all""")
con.execute(f"""create temp table a2 as select *, (app1 is null or app1='') as miss,
   (cc='US' and appl_year<=2011) as usold,
   (cc='CN' and appl_year between 2005 and 2009 and (app1 is null or app1='')) as cnmiss59,
   (case when app1 is null or app1='' then false else {corp_case} end) as corp from a""")
print('a', con.sql("select count(*) from a2").fetchone(), round(time.time() - t), flush=True)
con.execute("create temp table fa as select ksic, mg, cc, app1, min(year) fy from a2 where not miss group by all")
con.execute("create temp table fa1 as select ksic, mg, cc, app1, min(year) fy from a2 where not miss and not usold group by all")
con.execute("create temp table fac as select ksic, mg, cc, app1, min(year) fy from a2 where not miss and corp group by all")
con.execute("""create temp table p as
 select a2.ksic, a2.mg, a2.cc, a2.year, count(*) n, sum(miss::int) n_null, sum(usold::int) n_usold, sum(cnmiss59::int) n_cnmiss59, sum(corp::int) n_corp,
  count(distinct a2.app1) n_app,
  sum(case when fa.fy=a2.year then 1 else 0 end) n_newapp_apps,
  count(distinct case when fa.fy=a2.year then a2.app1 end) n_newapp,
  sum(case when not a2.usold and fa1.fy=a2.year then 1 else 0 end) n_newapp_apps_s1,
  count(distinct case when not a2.usold and fa1.fy=a2.year then a2.app1 end) n_newapp_s1,
  sum(case when a2.corp and fac.fy=a2.year then 1 else 0 end) n_newapp_apps_corp
 from a2 left join fa using(ksic,mg,cc,app1) left join fa1 using(ksic,mg,cc,app1) left join fac using(ksic,mg,cc,app1) group by all""")
con.execute(f"copy p to '{OUT}' (format parquet)")
print(con.sql(f"select cc, count(*), sum(n), sum(n_null), sum(n_usold), sum(n_cnmiss59), sum(n_corp), sum(n_newapp_apps), sum(n_newapp_apps_s1) from read_parquet('{OUT}') group by cc order by cc").df().to_string())
# 데이터 확인용: 국가×출원연도·공개연도 문서 단위 결측·법인형 비율(셀 dedupe 전 고유 출원 단위)
con.execute(f"""create temp table d as select cc, appl_no, min(year) py, min(appl_year) ay, any_value(app1) app1 from read_parquet('{B}') group by cc, appl_no""")
con.execute(f"""create temp table d2 as select *, (app1 is null or app1='') miss, (case when app1 is null or app1='' then null else {corp_case} end) corp from d""")
for yc, nm in [('ay', 'appl'), ('py', 'pub')]:
    con.execute(f"""copy (select cc, {yc} as year, count(*) n_docs, avg(miss::int) miss_rate, avg(corp::int) corp_rate from d2 group by 1,2 order by 1,2)
                    to '{ddb(os.path.join(CACHE, f'docrates_{which}_{nm}.csv'))}' (header)""")
# 공개연도별 US 문서의 출원연도 ≤2011 비중
con.execute(f"""copy (select py as year, count(*) n_us, avg((ay<=2011)::int) usold_sh from d2 where cc='US' group by 1 order by 1)
                to '{ddb(os.path.join(CACHE, f'us_old_by_pub_{which}.csv'))}' (header)""")
print('done', round(time.time() - t), flush=True)

# T2 패널 구축: 출원연도 2000–2018만 집계 (🔒 홀드아웃 목표연도 2019–2021 결과변수는 계산하지 않음)
import duckdb, os, time
T = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(T, '..', '_shared', 'meta', '*.parquet').replace(os.sep, '/')
C = os.path.join(T, 'cache').replace(os.sep, '/')
YMAX = 2018  # 절대 변경 금지(1단계): 검증 목표연도 상한
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
t0 = time.time()
con.sql(f"""create temp table b as
select cc, appl_no, appl_year::int y, ksic,
  nullif(trim(applicant),'') app,
  substr(replace(main_ipc,' ',''),1,4) sc,
  nullif(split_part(coalesce(prio_cc,''),'|',1),'') p1,
  case when cc='CN' and length(appl_no)>=12 and substr(appl_no,5,1)='2' then 1
       when cc='CN' and length(appl_no)=9 and substr(appl_no,3,1)='2' then 1
       when cc='KR' and substr(appl_no,1,2)='20' then 1 else 0 end um
from read_parquet('{P}') where appl_year between 2000 and {YMAX}""")
print('base', con.sql('select count(*) from b').fetchone(), round(time.time()-t0,1))
# 출원인 최초 등장연도(국가 내, 2000년부터 좌측절단)
con.sql("""create temp table fa as select cc, app, min(y) fy from b where app is not null group by 1,2""")
con.sql(f"""copy (select b.ksic, b.cc, b.y AS "year", count(*) n,
  sum(um) n_um,
  sum((p1 is not null and p1<>b.cc)::int) n_fprio,
  sum((p1 is not null)::int) n_prio,
  sum((fa.fy=b.y)::int) n_newapp,
  count(distinct b.app) n_app
 from b left join fa on b.cc=fa.cc and b.app=fa.app group by 1,2,3 order by 1,2,3)
 to '{C}/panel_kc.parquet' (format parquet)""")
con.sql(f"""copy (select ksic, cc, sc, y AS "year", count(*) n, sum(um) n_um from b group by 1,2,3,4)
 to '{C}/panel_kcs.parquet' (format parquet)""")
# 우선권 국가 흐름: (ksic, 출원국 cc, 우선권국 p1, year)
con.sql(f"""copy (select ksic, cc, p1, y AS "year", count(*) n from b where p1 is not null group by 1,2,3,4)
 to '{C}/panel_prio.parquet' (format parquet)""")
print('done', round(time.time()-t0,1))

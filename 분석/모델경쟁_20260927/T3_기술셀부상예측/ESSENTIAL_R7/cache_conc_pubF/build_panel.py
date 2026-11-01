import os
CACHE=os.environ.get("T3_CACHE","cache"); MAXYEAR=int(os.environ.get("T3_MAXYEAR","2018")); T0MAX=int(os.environ.get("T3_T0MAX","2015"))
os.makedirs(CACHE,exist_ok=True)
# 셀-국가-연도 패널 (2000~2018). 모든 '최초 등장' 계산은 해당 연도 이하 정보로 결정되는 값만 사용(첫 등장연도 == 해당 연도).
import duckdb, time
t=time.time()
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
con.execute(rf"""create view b as select ksic, cc, year, appl_no, mgn(ipc_raw) as mg, app1, prio_for, ipc_all, title
 from read_parquet('{CACHE}/base_rows.parquet') where mgn(ipc_raw) is not null""")
# 1) 출원 단위 dedupe (셀 내 동일 출원 중복 방지)
con.execute("create temp table a as select ksic, mg, cc, year, appl_no, any_value(app1) app1, max(prio_for) prio_for, any_value(ipc_all) ipc_all, any_value(title) title from b group by all")
print('a', con.sql("select count(*) from a").fetchone(), time.time()-t)
# 2) 출원인 첫 등장연도 (셀·국가 내)
con.execute("create temp table fa as select ksic, mg, cc, app1, min(year) fy from a where app1 is not null and app1<>'' group by all")
# 출원인 첫 등장 (KSIC·국가 내 — 산업 신규진입)
con.execute("create temp table fk as select ksic, cc, app1, min(year) fyk from a where app1 is not null and app1<>'' group by all")
con.execute("""create temp table p1 as
 select a.ksic, a.mg, a.cc, a.year, count(*) n, count(distinct a.app1) n_app, sum(prio_for) n_prio_for,
  sum(case when fa.fy=a.year then 1 else 0 end) n_newapp_apps,
  count(distinct case when fa.fy=a.year then a.app1 end) n_newapp,
  sum(case when fk.fyk=a.year then 1 else 0 end) n_newk_apps
 from a left join fa using(ksic,mg,cc,app1) left join fk using(ksic,cc,app1) group by all""")
print('p1', time.time()-t)
# HHI
con.execute("""create temp table h as select ksic,mg,cc,year, sum(s*s) hhi from (
  select ksic,mg,cc,year, count(*)*1.0/sum(count(*)) over (partition by ksic,mg,cc,year) s from a group by ksic,mg,cc,year,app1) group by all""")
# 3) IPC 공동분류 신규 조합: (ksic, mg, 다른 mg) 쌍의 첫 등장연도 (전 국가 통합)
con.execute(r"""create temp table pr as select distinct ksic, mg, cc, year, appl_no, mgn(x) mg2 from (
   select ksic, mg, cc, year, appl_no, unnest(string_split(ipc_all,'|')) x from a where ipc_all is not null) where mgn(x)<>mg and mgn(x) is not null""")
con.execute("create temp table fpr as select ksic, mg, mg2, min(year) fyp from pr group by all")
con.execute("""create temp table p2 as select ksic,mg,cc,year, count(distinct appl_no) n_multi, count(distinct case when fyp=year then appl_no end) n_newpair_apps,
   count(distinct case when fyp=year then mg2 end) n_newpair from pr join fpr using(ksic,mg,mg2) group by all""")
print('p2', time.time()-t)
# 4) 제목 신규 단어 (KR만, 공백 토큰, 2글자 이상)
con.execute(r"""create temp table tw as select distinct ksic, mg, year, appl_no, w from (
   select ksic, mg, year, appl_no, unnest(string_split(regexp_replace(lower(title),'[^0-9a-z가-힣 ]',' ','g'),' ')) w from a where cc='KR' and title is not null) where length(w)>=2""")
con.execute("create temp table fw as select ksic, mg, w, min(year) fyw from tw group by all")
con.execute("""create temp table p3 as select ksic,mg,'KR' cc,year, count(distinct case when fyw=year then appl_no end) n_newword_apps,
   count(distinct case when fyw=year then w end) n_newword from tw join fw using(ksic,mg,w) group by all""")
print('p3', time.time()-t)
con.execute(f"""copy (select p1.*, h.hhi, coalesce(p2.n_multi,0) n_multi, coalesce(p2.n_newpair_apps,0) n_newpair_apps, coalesce(p2.n_newpair,0) n_newpair,
   p3.n_newword_apps, p3.n_newword
  from p1 left join h using(ksic,mg,cc,year) left join p2 using(ksic,mg,cc,year) left join p3 using(ksic,mg,cc,year))
  to '{CACHE}/panel_ccy.parquet' (format parquet)""")
print(con.sql(f"select cc, count(*), sum(n) from read_parquet('{CACHE}/panel_ccy.parquet') group by cc").df())
print('done', time.time()-t)

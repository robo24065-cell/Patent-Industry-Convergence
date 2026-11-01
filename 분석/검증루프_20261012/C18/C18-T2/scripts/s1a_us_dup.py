# C18-T2 s1a: US 7/8자리 이중 수록 쌍 재현·일치율 (읽기 전용) → cache/us_dup_pairs.parquet, dup_pairs_check.csv, cache/s1a.json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
con = con3()
con.execute(f"""create table us as select appl_no, length(appl_no) L, lpad(appl_no,8,'0') a8, appl_date, appl_year, rgst_no, rgst_date, pub_date,
   title, applicant, main_ipc, ipc_all, main_cpc, family, prio_cc, prio_date, ksic::varchar ksic from read_parquet('{META}') where cc='US'""")
res = {}
res['n_us_rows'] = con.sql("select count(*), count(distinct appl_no), count(distinct a8) from us").fetchone()
res['len_dist'] = con.sql("select L, count(*) from us group by 1 order by 1").fetchall()
res['len7_series'] = con.sql("select substr(a8,1,2) s, count(*) from us where L=7 group by 1 order by 1").fetchall()
con.execute("""create table pr as select s.appl_no a7, l.appl_no a8,
  s.ksic k7, l.ksic k8, s.title t7, l.title t8, s.appl_date d7, l.appl_date d8, s.appl_year y7, l.appl_year y8,
  s.main_ipc i7, l.main_ipc i8, s.ipc_all ia7, l.ipc_all ia8, s.pub_date p7, l.pub_date p8, s.rgst_no r7, l.rgst_no r8,
  s.rgst_date rd7, l.rgst_date rd8, s.family f7, l.family f8, s.applicant ap7, l.applicant ap8, s.prio_cc pc7, l.prio_cc pc8, s.main_cpc c7, l.main_cpc c8
  from us s join us l on s.L=7 and l.L=8 and s.a8=l.appl_no""")
res['n_pairs'] = con.sql("select count(*), count(distinct a7), count(distinct a8) from pr").fetchone()
res['n_len7_without_twin'] = con.sql("select count(*) from us where L=7 and a8 not in (select appl_no from us where L=8)").fetchone()[0]
con.execute(r"""create macro nt(x) as regexp_replace(lower(coalesce(x,'')),'\s+',' ','g')""")
con.execute(r"""create macro sset(x) as list_sort(list_distinct(string_split(coalesce(x,''),'|')))""")
eq = lambda a, b: f"avg(case when {a} is null and {b} is null then 1 when {a}={b} then 1 else 0 end)"
q = f"""select count(*) n,
  avg((nt(t7)=nt(t8))::int) title_eq, {eq('d7','d8')} appl_date_eq, {eq('i7','i8')} main_ipc_eq,
  avg((sset(ia7)=sset(ia8))::int) ipc_set_eq, avg((k7=k8)::int) ksic_eq, {eq('p7','p8')} pub_date_eq,
  {eq('r7','r8')} rgst_eq, avg((sset(f7)=sset(f8))::int) family_set_eq, {eq('ap7','ap8')} applicant_eq, {eq('pc7','pc8')} prio_eq, {eq('c7','c8')} cpc_eq,
  avg((t7 is null or t8 is null)::int) title_any_null, avg((p7 is null)::int) pub7_null, avg((p8 is null)::int) pub8_null,
  avg((r7 is not null)::int) r7_filled, avg((r8 is not null)::int) r8_filled
  from pr"""
d = con.sql(q).df()
res['match_rates_all'] = d.round(5).to_dict('records')[0]
res['match_rates_ksic_diff'] = con.sql(q + " where k7<>k8").df().round(5).to_dict('records')[0]
res['same_patent_rule'] = bool(res['match_rates_all']['title_eq'] >= 0.95 and res['match_rates_all']['appl_date_eq'] >= 0.95)
res['by_appl_year'] = con.sql("select y8, count(*) from pr group by 1 order by 1").fetchall()
res['by_pub_year'] = con.sql("select substr(p8,1,4) y, count(*) from pr group by 1 order by 1").fetchall()
res['by_series'] = con.sql("select substr(a8,1,2) s, count(*) from pr group by 1 order by 1").fetchall()
res['by_ksic8_top'] = con.sql("select k8, count(*) n from pr group by 1 order by 2 desc limit 15").fetchall()
res['n_ksic8'] = con.sql("select count(distinct k8) from pr").fetchone()[0]
res['ksic_share_vs_us'] = con.sql("""with a as (select k8 k, count(*) n from pr group by 1), b as (select ksic k, count(*) m from us group by 1)
   select max(n*1.0/m), median(n*1.0/m), sum(n)*1.0/sum(m) from b left join a using(k) """).fetchone()
# 쌍 중 하나가 제목·출원일 다를 때 예시
res['mismatch_examples'] = con.sql("select a7, a8, k7, k8, left(t7,60), left(t8,60), d7, d8 from pr where nt(t7)<>nt(t8) or coalesce(d7,'')<>coalesce(d8,'') limit 10").fetchall()
# 행 단위 표(dup_pairs_check.csv): 모든 쌍의 일치 플래그 + KSIC 다른 쌍은 내용
con.execute(f"""copy (select a7, a8, k7, k8, (k7=k8)::int ksic_eq, (nt(t7)=nt(t8))::int title_eq, (coalesce(d7,'')=coalesce(d8,''))::int appl_date_eq,
   (coalesce(i7,'')=coalesce(i8,''))::int main_ipc_eq, (sset(ia7)=sset(ia8))::int ipc_set_eq, (coalesce(p7,'')=coalesce(p8,''))::int pub_date_eq,
   (sset(f7)=sset(f8))::int family_set_eq, (coalesce(ap7,'')=coalesce(ap8,''))::int applicant_eq, y8 appl_year, substr(p8,1,4) pub_year, substr(a8,1,2) series,
   case when k7<>k8 then left(t7,80) end title7, case when k7<>k8 then left(t8,80) end title8, case when k7<>k8 then i7 end ipc7, case when k7<>k8 then i8 end ipc8
   from pr order by ksic_eq, a8) to '{ddb(W)}/dup_pairs_check.csv' (header, delimiter ',')""")
con.execute(f"copy (select a7, a8, k7, k8, y8, substr(p8,1,4) py, i8, (nt(t7)=nt(t8))::int title_eq from pr) to '{ddb(C)}/us_dup_pairs.parquet' (format parquet)")
jdump(res, 's1a.json')
out(json.dumps(res, ensure_ascii=False, default=str)[:5000])

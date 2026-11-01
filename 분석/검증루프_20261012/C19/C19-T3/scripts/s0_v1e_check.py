# C19-T3 s0: C18-T2 meta_V1E.parquet 검증(PREREG 1절) + data_check (a)-(d) 수치. 독립 쌍 키로 다시 찾는다(duckdb, 읽기 전용).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
con = con3()
V1E = ddb(C18T2 + r'\cache\meta_V1E.parquet')
res = {}
res['rows'] = {k: con.sql(f"select cc, count(*) from read_parquet('{p}') group by 1 order by 1").fetchall() for k, p in [('P0', META), ('V1E', V1E)]}
res['schema_same'] = [tuple(x[:2]) for x in con.sql(f"describe select * from read_parquet('{V1E}')").fetchall()] == [tuple(x[:2]) for x in con.sql(f"describe select * from read_parquet('{META}')").fetchall()]
# 독립 쌍: US 7자리 → lpad 8 이 있는 8자리; EP 앞 0 제거 키가 같은 서로 다른 번호(길이 다름)
con.execute(f"create table us as select appl_no, ksic::varchar ksic, appl_year, pub_date, rgst_date, title from read_parquet('{META}') where cc='US'")
con.execute(f"create table ep as select appl_no, ksic::varchar ksic, appl_year, pub_date, rgst_date, title from read_parquet('{META}') where cc='EP'")
con.execute("create table us_drop as select s.* from us s join us l on length(s.appl_no)=7 and length(l.appl_no)=8 and lpad(s.appl_no,8,'0')=l.appl_no")
con.execute("create table ep_k as select *, ltrim(appl_no,'0') k from ep")
con.execute("create table ep_drop as select s.* from ep_k s join ep_k l on s.k=l.k and length(s.appl_no)<length(l.appl_no)")
res['indep_pairs'] = dict(us=con.sql("select count(*), count(distinct appl_no) from us_drop").fetchone(), ep=con.sql("select count(*), count(distinct appl_no) from ep_drop").fetchone(),
                          ep_groups_gt2=con.sql("select count(*) from (select k from ep_k group by k having count(distinct appl_no)>2)").fetchone()[0],
                          ep_lens=con.sql("select length(appl_no), count(*) from ep_drop group by 1 order by 1").fetchall())
# P0 - V1E 차집합(행 단위, cc+appl_no+ksic 다중집합)
con.execute(f"""create table diff as with a as (select cc::varchar cc, appl_no, ksic::varchar ksic, count(*) n from read_parquet('{META}') where cc in ('US','EP') group by all),
  b as (select cc::varchar cc, appl_no, ksic::varchar ksic, count(*) m from read_parquet('{V1E}') where cc in ('US','EP') group by all)
  select a.cc, a.appl_no, a.ksic, a.n - coalesce(b.m,0) d from a left join b using(cc, appl_no, ksic) where a.n <> coalesce(b.m,0)""")
res['diff'] = con.sql("select cc, count(*), sum(d), min(d), max(d) from diff group by 1").fetchall()
res['diff_eq_indep'] = dict(
    us_in_diff_not_indep=con.sql("select count(*) from diff where cc='US' and appl_no not in (select appl_no from us_drop)").fetchone()[0],
    us_indep_not_in_diff=con.sql("select count(*) from us_drop where appl_no not in (select appl_no from diff where cc='US')").fetchone()[0],
    ep_in_diff_not_indep=con.sql("select count(*) from diff where cc='EP' and appl_no not in (select appl_no from ep_drop)").fetchone()[0],
    ep_indep_not_in_diff=con.sql("select count(*) from ep_drop where appl_no not in (select appl_no from diff where cc='EP')").fetchone()[0])
res['extra_rows_in_V1E'] = con.sql(f"select count(*) from (select cc, appl_no, ksic from read_parquet('{V1E}') except all select cc, appl_no, ksic from read_parquet('{META}'))").fetchone()[0]
# (b) 분포: 제거 행의 출원·공개·등록 연도, KSIC
for nm in ['us_drop', 'ep_drop']:
    res[nm + '_appl_year'] = con.sql(f"select appl_year, count(*) from {nm} group by 1 order by 1").fetchall()
    res[nm + '_pub_year'] = con.sql(f"select substr(pub_date,1,4), count(*) from {nm} group by 1 order by 1").fetchall()
    res[nm + '_least_year'] = con.sql(f"select substr(least(coalesce(pub_date,'9999'),coalesce(rgst_date,'9999')),1,4) y, count(*) from {nm} group by 1 order by 1").fetchall()
    res[nm + '_rgst_year'] = con.sql(f"select substr(rgst_date,1,4), count(*) from {nm} group by 1 order by 1").fetchall()
    res[nm + '_ksic_top'] = con.sql(f"select ksic, count(*) from {nm} group by 1 order by 2 desc limit 10").fetchall()
    res[nm + '_n_ksic'] = con.sql(f"select count(distinct ksic) from {nm}").fetchone()[0]
# (c) 남는 짝(긴 형식) 존재: 제거 행마다 같은 KSIC의 짝이 V1E에 남는지
res['partner_same_ksic'] = dict(
    us=con.sql("select avg((exists(select 1 from us l where length(l.appl_no)=8 and l.appl_no=lpad(s.appl_no,8,'0') and l.ksic=s.ksic))::int) from us_drop s").fetchone()[0],
    ep=con.sql("select avg((exists(select 1 from ep_k l where l.k=s.k and length(l.appl_no)>length(s.appl_no) and l.ksic=s.ksic))::int) from ep_drop s").fetchone()[0])
# 연도별 비중(그 해 국가 행 대비)
res['us_drop_share_by_ay'] = con.sql("select u.appl_year, count(d.appl_no)*1.0/count(*) from us u left join us_drop d on u.appl_no=d.appl_no and u.ksic=d.ksic where u.appl_year between 1999 and 2003 group by 1 order by 1").fetchall()
res['ep_drop_share_by_ay'] = con.sql("select u.appl_year, round(count(d.appl_no)*1.0/count(*),4) from ep u left join ep_drop d on u.appl_no=d.appl_no and u.ksic=d.ksic where u.appl_year between 1999 and 2011 group by 1 order by 1").fetchall()
jdump(res, 's0_v1e_check.json'); out(json.dumps(res, ensure_ascii=False, default=str)[:6000])

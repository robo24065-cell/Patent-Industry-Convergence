# C18-T2 s1c: EP·JP 정규화 키 충돌 쌍의 일치율(같은 문서인지) — US D1과 같은 방식 → cache/s1c.json, cache/other_dup_pairs.parquet
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
con = con3()
con.execute(f"""create table m as select cc::varchar cc, appl_no, length(appl_no) L, appl_date, appl_year, rgst_no, rgst_date, pub_date, title, applicant,
   main_ipc, ipc_all, family, prio_cc::varchar prio_cc, ksic::varchar ksic, ltrim(regexp_replace(appl_no,'[^0-9]','','g'),'0') k
   from read_parquet('{META}') where cc in ('EP','JP','KR','CN')""")
con.execute("create table g as select cc, k from m group by 1,2 having count(distinct appl_no)>1")
con.execute("""create table pr as select a.cc, a.k, a.appl_no n1, b.appl_no n2, a.L L1, b.L L2, a.ksic k1, b.ksic k2, a.title t1, b.title t2, a.appl_date d1, b.appl_date d2,
   a.appl_year y1, b.appl_year y2, a.main_ipc i1, b.main_ipc i2, a.ipc_all ia1, b.ipc_all ia2, a.pub_date p1, b.pub_date p2, a.rgst_no r1, b.rgst_no r2,
   a.family f1, b.family f2, a.applicant ap1, b.applicant ap2
   from m a join g using(cc,k) join m b on a.cc=b.cc and a.k=b.k and a.appl_no<b.appl_no""")
con.execute(r"""create macro nt(x) as regexp_replace(lower(coalesce(x,'')),'\s+',' ','g')""")
con.execute(r"""create macro sset(x) as list_sort(list_distinct(string_split(coalesce(x,''),'|')))""")
eq = lambda a, b: f"avg(case when {a} is null and {b} is null then 1 when {a}={b} then 1 else 0 end)"
q = f"""select cc, count(*) n, count(distinct k) keys, avg((nt(t1)=nt(t2))::int) title_eq, {eq('d1','d2')} appl_date_eq, {eq('i1','i2')} main_ipc_eq,
  avg((sset(ia1)=sset(ia2))::int) ipc_set_eq, avg((k1=k2)::int) ksic_eq, {eq('p1','p2')} pub_date_eq, {eq('r1','r2')} rgst_eq,
  avg((sset(f1)=sset(f2))::int) family_set_eq, {eq('ap1','ap2')} applicant_eq from pr group by cc order by cc"""
res = {'pairs': con.sql(q).df().round(5).to_dict('records')}
res['len_combo'] = con.sql("select cc, L1, L2, count(*) from pr group by all order by all").fetchall()
res['jp_by_suffix'] = con.sql(f"""select (n1 like '%U' or n2 like '%U') has_u, count(*) n, avg((nt(t1)=nt(t2))::int) title_eq, {eq('d1','d2')} date_eq, {eq('i1','i2')} ipc_eq
   from pr where cc='JP' group by 1""").fetchall()
res['by_year'] = con.sql("select cc, least(y1,y2) y, count(*) from pr group by 1,2 order by 1,2").fetchall()
res['examples'] = con.sql("select cc, n1, n2, k1, k2, left(t1,40), left(t2,40), d1, d2, p1, p2 from pr using sample 12 rows").fetchall()
res['ep_same_ksic'] = con.sql("select avg((k1=k2)::int), count(*) from pr where cc='EP'").fetchall()
res['n_rows_cc'] = con.sql("select cc, count(*) from m group by 1 order by 1").fetchall()
con.execute(f"""copy (select cc, k, n1, n2, L1, L2, k1, k2, y1, y2, substr(p1,1,4) py1, substr(p2,1,4) py2, (nt(t1)=nt(t2))::int title_eq,
   (coalesce(d1,'')=coalesce(d2,''))::int date_eq, i1, i2 from pr) to '{ddb(C)}/other_dup_pairs.parquet' (format parquet)""")
jdump(res, 's1c.json')
out(json.dumps(res, ensure_ascii=False, default=str)[:5000])

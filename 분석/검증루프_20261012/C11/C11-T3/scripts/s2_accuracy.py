# -*- coding: utf-8 -*-
"""s2 (2) 연결 정확도: 공유 r3ros_family_us_pairs(=T3·N5가 쓰는 파일) 쌍 표본에서 M1(역방향 family)·M2(우선일 정합)·M3(IPC 서브클래스)·M4(날짜 필요조건)
 + 우연 기준선(같은 국가 내 P 순열, 시드 20261013) + 인접 번호 기준선(P+1).
정의(사전 고정):
 U(P) = 제공 US 레코드 u (u != k) 중 family 코드키 토큰 값 = P 또는 rgst_no = P.
 M1a(KR만) = U(P)의 어떤 u.family 에 k의 KR 자기 토큰('KR'+공개번호13, 'KR'+공개번호11(앞 '10' 제거), 'KR'+등록번호 앞 9자리, 'KR'+출원번호13) 이 있다.
 M1b = k.family 와 U(P) 어떤 u.family 의 비US 토큰 교집합 >=1.  M1 = M1a or M1b.
 M2  = k의 날짜 집합(prio_date 전부 + appl_date) 과 u의 날짜 집합(prio_date 전부 + appl_date) 교집합 != 공집합 (어떤 u든).
 M3  = k main_ipc 서브클래스(앞 4자) 가 U(P) ipc_all 문자열 안에 있음.
 M4  = P 발행일(r3ros_patent_sci) >= k 최초일(min(prio, appl)), KPSS filing_date 있으면 >= k 최초일 - 365일.
 정밀도 = 평가 가능(U(P) 비어있지 않음) 쌍 중 (M1 or M2).
"""
import json
from common import *
import pandas as pd
c = con()
SEED = 20261013
REC = f"read_parquet('{W}/cache/rec.parquet')"
c.execute(f"""create table pr as select p.cc, p.appl_no, p.ksic, p.us_pn, p.src, r.appl_year, r.appl_date, r.prio_date, r.main_ipc
   from read_parquet('{EXT}/r3ros_family_us_pairs.parquet') p join {REC} r using(cc, appl_no, ksic)""")
c.execute(f"create table ps as select us_pn, issue_date::date idt, issue_year from read_parquet('{EXT}/r3ros_patent_sci.parquet')")
c.execute(f"create table kp as select patent_num::bigint pn, filing_date::date fdt from read_parquet('{EXT}/quality_kpss_2025.parquet')")
R = {}
R['pairs_all'] = c.sql("select src, cc, count(*) from pr group by 1,2 order by 1,2").fetchall()
log('pairs', R['pairs_all'])
c.execute(f"""create table smp as select * from (select *, row_number() over (partition by cc order by hash(appl_no||'#'||us_pn::varchar||'{SEED}')) rn
   from pr where src='family') where rn <= {int(os.environ.get('NS','30000'))}""")
c.execute(f"""create table perm as select a.cc, a.appl_no, a.ksic, a.appl_year, a.appl_date, a.prio_date, a.main_ipc, b.us_pn, 'perm' kind
   from (select *, row_number() over (partition by cc order by rn) i from smp) a
   join (select cc, us_pn, row_number() over (partition by cc order by hash(us_pn::varchar||'{SEED}x')) i from smp) b using(cc, i)""")
c.execute("""create table ev as select cc, appl_no, ksic, appl_year, appl_date, prio_date, main_ipc, us_pn, 'link' kind from smp
   union all select * from perm
   union all select cc, appl_no, ksic, appl_year, appl_date, prio_date, main_ipc, us_pn+1, 'next' from smp""")
c.execute(f"""create table um as select distinct u_appl, P from (
   select appl_no u_appl, num P from read_parquet('{W}/cache/tk.parquet') where cc='US' and code_key
   union all select appl_no, try_cast(rgst_no as bigint) from {REC} where cc='US' and regexp_full_match(coalesce(rgst_no,''),'[0-9]{{7,8}}'))
   where P in (select us_pn from ev)""")
c.execute("create table um2 as select u_appl, P from (select *, row_number() over (partition by P order by hash(u_appl)) r from um) where r <= 12")
c.execute("drop table um; alter table um2 rename to um")
log('um', c.sql("select count(*), count(distinct P) from um").fetchone())
c.execute("create table need as select distinct cc, appl_no from ev union select 'US', u_appl from um")
c.execute(f"""create table fams as select m.cc, m.appl_no, any_value(m.family) fam from read_parquet('{META}') m
   semi join need n on n.cc=m.cc and n.appl_no=m.appl_no group by 1,2""")
log('fams', c.sql("select count(*) from fams").fetchone())
c.execute(f"""create table urec as select r.appl_no u_appl, r.appl_date u_ad, r.prio_date u_pd, r.ipc_all u_ipc, r.main_ipc u_main, f.fam u_fam
   from {REC} r join fams f on f.cc='US' and f.appl_no=r.appl_no semi join (select distinct u_appl from um) q on q.u_appl=r.appl_no where r.cc='US'""")
log('urec')
c.execute(f"""create table krid as select r.appl_no, list_distinct(list_filter(['KR'||coalesce(trim(k.공개번호),''),
     'KR'||case when length(trim(k.공개번호))=13 then substr(trim(k.공개번호),3) else '' end,
     'KR'||left(coalesce(r.rgst_no,''),9), 'KR'||r.appl_no], x -> length(x) > 4)) ids
   from {REC} r left join read_parquet('{KRP}') k on k.출원번호=r.appl_no where r.cc='KR' and r.appl_no in (select appl_no from ev where cc='KR')""")
log('krid', c.sql("select count(*) from krid").fetchone())
c.execute("""create macro dset(pd, ad) as list_distinct(list_filter(list_concat(string_split(coalesce(pd,''),'|'), [coalesce(ad,'')]), x -> length(x)=8))""")
c.execute("""create macro nonus(f) as list_distinct(list_filter(string_split(coalesce(f,''),'|'), x -> length(x)>3 and left(x,2)<>'US'))""")
c.execute("""create table eu as select e.*, case when e.cc='US' and um.u_appl=e.appl_no then null else um.u_appl end u_appl
   from ev e left join um on um.P=e.us_pn""")
c.execute("""create table j as select e.kind, e.cc, e.appl_no, e.us_pn, e.appl_year, e.main_ipc, e.prio_date, e.appl_date,
     u.u_appl, u.u_ipc,
     coalesce(len(list_intersect(nonus(fk.fam), nonus(u.u_fam)))>0, false) m1b,
     coalesce(len(list_intersect(kid.ids, string_split(coalesce(u.u_fam,''),'|')))>0, false) m1a,
     coalesce(len(list_intersect(dset(e.prio_date, e.appl_date), dset(u.u_pd, u.u_ad)))>0, false) m2
   from eu e left join fams fk on fk.cc=e.cc and fk.appl_no=e.appl_no
   left join krid kid on kid.appl_no=e.appl_no and e.cc='KR'
   left join urec u on u.u_appl=e.u_appl""")
log('j', c.sql("select count(*) from j").fetchone())
c.execute("""create table pj as select kind, cc, appl_no, us_pn, any_value(appl_year) appl_year, any_value(main_ipc) main_ipc,
     any_value(prio_date) prio_date, any_value(appl_date) appl_date,
     count(u_appl) n_u, coalesce(bool_or(m1a),false) m1a, coalesce(bool_or(m1b),false) m1b, coalesce(bool_or(m2),false) m2,
     coalesce(bool_or(length(coalesce(main_ipc,''))>=4 and position(left(replace(coalesce(main_ipc,''),' ',''),4) in replace(coalesce(u_ipc,''),' ',''))>0), false) m3
   from j group by 1,2,3,4""")
c.execute("""create table pf as select p.*, ps.idt, ps.issue_year, kp.fdt,
     list_min(list_transform(dset(p.prio_date, p.appl_date), x -> strptime(x,'%Y%m%d')::date)) kmin
   from pj p left join ps on ps.us_pn=p.us_pn left join kp on kp.pn=p.us_pn""")
c.execute("""create table fin as select *, (m1a or m1b) m1, ((m1a or m1b) or m2) ok,
   (idt is not null and kmin is not null and idt >= kmin and (fdt is null or fdt >= kmin - interval 365 day)) m4,
   (appl_year <= 2019 and issue_year between 2001 and 2021) t3win from pf""")
c.execute(f"copy fin to '{W}/cache/s2_fin.parquet' (format parquet)")
def summ(where, label):
    d = c.sql(f"""select kind, cc, count(*) n, avg((n_u>0)::int) evaluable,
        avg(case when n_u>0 then ok::int end) prec, avg(case when n_u>0 then m1::int end) m1,
        avg(case when n_u>0 then m1a::int end) m1a_kr, avg(case when n_u>0 then m1b::int end) m1b,
        avg(case when n_u>0 then m2::int end) m2, avg(case when n_u>0 then m3::int end) m3,
        avg(case when kmin is not null and idt is not null then m4::int end) m4, sum((n_u>0)::int) n_eval
        from fin where {where} group by rollup(kind, cc) order by 1,2""").df()
    d.insert(0, 'subset', label); return d
A = pd.concat([summ("true", "all_family_pairs"), summ("t3win", "T3_window"), summ("t3win and cc<>'US'", "T3_window_nonUS")])
A = A[A.kind.notna()]
A.to_csv(W + "/link_audit.csv", index=False, encoding="utf-8-sig")
print(A.round(4).to_string())
R['t3_nonus'] = c.sql("""select kind, count(*) n, sum((n_u>0)::int) n_eval, avg(case when n_u>0 then ok::int end) prec,
   avg(case when n_u>0 then m3::int end) m3, avg(case when kmin is not null and idt is not null then m4::int end) m4 from fin where t3win and cc<>'US' group by 1 order by 1""").fetchall()
log('t3_nonus', R['t3_nonus'])
R['fail_profile'] = c.sql("""select ok, count(*), avg(m3::int), avg(m4::int), avg(appl_year), avg(n_u) from fin where kind='link' and t3win and cc<>'US' and n_u>0 group by 1""").fetchall()
log('fail', R['fail_profile'])
json.dump(R, open(W + "/res/s2_accuracy.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log('done')

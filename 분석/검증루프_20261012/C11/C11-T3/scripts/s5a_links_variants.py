# -*- coding: utf-8 -*-
"""s5a: 민감도용 수정 연결 V1·V2 (PREREG 변경 기록 1) + 셀×발행연도 표 재구성(V0=공식 재현, V1, V2).
입력(읽기만): _shared/ext/r3ros_work/fam_us.parquet·rec.parquet, r3ros_patent_sci.parquet, quality_kpss_2025.parquet, _shared/meta, cache/tk.parquet·rec.parquet(이 폴더)
V2: 같은 우선일 연결만. k 날짜 집합(prio 전부 + appl_date) 과 U(P)\{k} 어떤 레코드 날짜 집합이 겹치면 유지.
    U(P)\{k} 가 비면: KPSS filing_date 가 있으면 k 날짜 집합에 있을 때만 유지, 없으면 유지(판정 불가). src='own' 은 유지.
V1: 레코드당 연결 US 특허 수(us_pn 3M–13M) ≤10 인 레코드의 연결만.
출력: cache/fam_us_V1.parquet, cache/fam_us_V2.parquet, cache/ros_cell_{V0,V1,V2}.parquet, res/s5a_links.json
"""
import json
from common import *
c = con()
R = {}
FAM = f"read_parquet('{EXT}/r3ros_work/fam_us.parquet')"
c.execute(f"create table fam as select * from {FAM} where us_pn between 3000000 and 13000000")
R['fam_rows'] = c.sql("select src, count(*) from fam group by 1").fetchall()
# k 날짜
c.execute(f"""create table kd as select distinct r.cc, r.appl_no, d from (select cc, appl_no, unnest(list_concat(string_split(coalesce(prio_date,''),'|'), [coalesce(appl_date,'')])) d
   from read_parquet('{W}/cache/rec.parquet') r semi join (select distinct cc, appl_no from fam) f using(cc, appl_no)) r where length(d)=8""")
log('kd', c.sql("select count(*) from kd").fetchone())
# U(P): US 레코드 ↔ P
c.execute(f"""create table um as select distinct u_appl, P from (
   select appl_no u_appl, num P from read_parquet('{W}/cache/tk.parquet') where cc='US' and code_key
   union all select appl_no, try_cast(rgst_no as bigint) from read_parquet('{W}/cache/rec.parquet') where cc='US' and regexp_full_match(coalesce(rgst_no,''),'[0-9]{{7,8}}'))
   where P between 3000000 and 13000000""")
c.execute(f"""create table ud as select distinct appl_no u_appl, d from (select appl_no, unnest(list_concat(string_split(coalesce(prio_date,''),'|'), [coalesce(appl_date,'')])) d
   from read_parquet('{W}/cache/rec.parquet') where cc='US') where length(d)=8""")
c.execute("create table upd as select distinct um.P, um.u_appl, ud.d from um join ud using(u_appl)")
log('upd', c.sql("select count(*) from upd").fetchone())
# 쌍별 판정
c.execute("""create table pair_match as select distinct f.cc, f.appl_no, f.us_pn from fam f join kd using(cc, appl_no)
   join upd u on u.P=f.us_pn and u.d=kd.d where not (f.cc='US' and u.u_appl=f.appl_no)""")
c.execute("""create table pair_hasu as select distinct f.cc, f.appl_no, f.us_pn from fam f join um on um.P=f.us_pn where not (f.cc='US' and um.u_appl=f.appl_no)""")
c.execute(f"create table kp as select patent_num::bigint pn, strftime(filing_date::date,'%Y%m%d') fd from read_parquet('{EXT}/quality_kpss_2025.parquet')")
c.execute("""create table fam2 as select f.*, (m.us_pn is not null) dmatch, (h.us_pn is not null) hasu, (kp.pn is not null) haskp,
     exists(select 1 from kd where kd.cc=f.cc and kd.appl_no=f.appl_no and kd.d=kp.fd) kpmatch
   from fam f left join pair_match m using(cc, appl_no, us_pn) left join pair_hasu h using(cc, appl_no, us_pn) left join kp on kp.pn=f.us_pn""")
c.execute("""create table fam2b as select *, case when src='own' then true when hasu then dmatch when haskp then kpmatch else true end keep_v2 from fam2""")
R['v2_breakdown'] = c.sql("""select src, cc, count(*) n, avg(hasu::int) hasu, avg(case when hasu then dmatch::int end) dmatch_if_u,
   avg((not hasu and haskp)::int) kp_only, avg(case when not hasu and haskp then kpmatch::int end) kpmatch_if_kponly,
   avg((not hasu and not haskp)::int) undecidable, avg(keep_v2::int) keep from fam2b group by rollup(src, cc) order by 1,2""").fetchall()
for x in R['v2_breakdown']: print(x)
c.execute(f"copy (select cc, appl_no, ksic, us_pn, src from fam2b where keep_v2) to '{W}/cache/fam_us_V2.parquet' (format parquet)")
# V1
c.execute("create table nrec as select cc, appl_no, ksic, count(distinct us_pn) n_us from fam group by 1,2,3")
R['v1_nrec_dist'] = c.sql("select count(*), avg((n_us<=10)::int), quantile_cont(n_us,[0.5,0.9,0.99]), max(n_us) from nrec").fetchone()
R['v1_pairs_share_from_big'] = c.sql("select sum(case when n.n_us>10 then 1 else 0 end)/count(*) from fam f join nrec n using(cc, appl_no, ksic)").fetchone()
c.execute(f"copy (select f.* from fam f join nrec n using(cc, appl_no, ksic) where n.n_us <= 10) to '{W}/cache/fam_us_V1.parquet' (format parquet)")
log('v1', R['v1_nrec_dist'], R['v1_pairs_share_from_big'])
# 셀×발행연도 표 (build_ros_pubrule_r3d.py 와 같은 집계)
c.execute(f"""create table pubrec as select cc, appl_no, ksic,
   min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
   from read_parquet('{META}') group by 1,2,3""")
for V, src in (('V0', FAM), ('V1', f"read_parquet('{W}/cache/fam_us_V1.parquet')"), ('V2', f"read_parquet('{W}/cache/fam_us_V2.parquet')")):
    c.execute(f"""create or replace table cp as select r.ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year, min(p.pub_year) first_pub_year
       from {src} f join read_parquet('{EXT}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
       left join pubrec p using(cc, appl_no, ksic) where r.mg is not null group by 1,2,3""")
    c.execute(f"""copy (select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
       count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
       from cp c join read_parquet('{EXT}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all)
       to '{W}/cache/ros_cell_{V}.parquet' (format parquet)""")
    R[f'cell_{V}'] = c.sql(f"select count(*), sum(n_us_pat), sum(n_us_pat_sci), sum(n_paper_cit) from read_parquet('{W}/cache/ros_cell_{V}.parquet')").fetchone()
    log(V, R[f'cell_{V}'])
# V0 = 공식 표 재현 대조
OFF = f"read_parquet('{T3R3}/ros_cell_issueyear_pubrule.parquet')"
R['V0_vs_official'] = c.sql(f"""select count(*) n, sum((a.n_us_pat is distinct from b.n_us_pat or a.n_us_pat_sci is distinct from b.n_us_pat_sci or a.n_paper_cit is distinct from b.n_paper_cit)::int) diff
   from (select ksic, mg, issue_year, first_rec_year, first_pub_year, sum(n_us_pat) n_us_pat, sum(n_us_pat_sci) n_us_pat_sci, sum(n_paper_cit) n_paper_cit from read_parquet('{W}/cache/ros_cell_V0.parquet') group by all) a
   full join (select ksic, mg, issue_year, first_rec_year, first_pub_year, sum(n_us_pat) n_us_pat, sum(n_us_pat_sci) n_us_pat_sci, sum(n_paper_cit) n_paper_cit from {OFF} group by all) b
   using(ksic, mg, issue_year, first_rec_year, first_pub_year)""").fetchone()
log('V0 vs official', R['V0_vs_official'])
json.dump(R, open(W + "/res/s5a_links.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log('done')

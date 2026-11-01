# -*- coding: utf-8 -*-
"""s2b: 수동 판정용 무작위 100쌍(T3 창, 비US, 평가 가능, 시드 고정). M 지표는 별도 파일(manual_key.csv)로 분리해 판정 전에 보지 않는다.
제목은 제공 데이터라 이 폴더 cache 안에만 둔다(반출 금지)."""
from common import *
c = con()
REC = f"read_parquet('{W}/cache/rec.parquet')"
c.execute(f"""create table s as select * from read_parquet('{W}/cache/s2_fin.parquet')
   where kind='link' and t3win and cc<>'US' and n_u>0 order by hash(appl_no||us_pn::varchar||'manual20261013') limit 100""")
c.execute(f"""create table um as select distinct appl_no u_appl, num P from read_parquet('{W}/cache/tk.parquet') where cc='US' and code_key and num in (select us_pn from s)""")
c.execute(f"""create table t as select cc, appl_no, any_value(title) title from read_parquet('{META}') m semi join
   (select cc, appl_no from s union select 'US', u_appl from um) q using(cc, appl_no) group by 1,2""")
c.execute(f"create table kp as select patent_num::bigint pn, strftime(filing_date::date,'%Y%m%d') fd from read_parquet('{EXT}/quality_kpss_2025.parquet') where patent_num in (select us_pn from s)")
c.execute(f"""create table nus as select appl_no u_appl, count(*) n_usreg from read_parquet('{W}/cache/tk.parquet') where cc='US' and code_key and appl_no in (select u_appl from um) group by 1""")
c.execute(f"""create table u as select um.P, r.appl_no u_appl, r.appl_date, r.prio_date, r.main_ipc, t.title,
     case when try_cast(r.rgst_no as bigint)=um.P then 0 when r.appl_date=kp.fd then 1 else 2 end pri, n.n_usreg,
     case when try_cast(r.rgst_no as bigint)=um.P then 'rgst=P' when r.appl_date=kp.fd then 'KPSS출원일' else 'famUS'||n.n_usreg::varchar end tag
   from um join {REC} r on r.cc='US' and r.appl_no=um.u_appl join t on t.cc='US' and t.appl_no=r.appl_no
   left join kp on kp.pn=um.P left join nus n on n.u_appl=um.u_appl""")
c.execute(f"""create table out as select row_number() over (order by s.cc, s.appl_no) id, s.cc, s.appl_no, s.us_pn, r.appl_date k_appl, r.prio_date k_prio,
   r.main_ipc k_ipc, t.title k_title, s.issue_year,
   (select string_agg('['||tag||'] '||coalesce(main_ipc,'')||' / '||coalesce(appl_date,'')||' / prio '||left(coalesce(prio_date,''),40)||' / '||left(coalesce(title,''),100), ' || ')
      from (select * from u where u.P=s.us_pn order by pri, n_usreg, u_appl limit 2)) us_side
   from s join {REC} r using(cc, appl_no) left join t using(cc, appl_no)""")
c.execute(f"copy (select id, cc, us_pn, k_appl, left(coalesce(k_prio,''),40) k_prio, k_ipc, k_title, issue_year, us_side from out order by id) to '{W}/cache/manual_sheet.csv' (header, delimiter ',')")
c.execute(f"""copy (select o.id, f.m1, f.m2, f.m3, f.m4, f.ok from out o join s f on f.appl_no=o.appl_no and f.us_pn=o.us_pn and f.cc=o.cc order by id)
   to '{W}/cache/manual_key.csv' (header, delimiter ',')""")
log('done', c.sql("select cc, count(*) from out group by 1").fetchall())

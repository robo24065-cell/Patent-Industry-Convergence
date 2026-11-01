# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5\m1b_field_counts.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""map5 1b: KSIC×국가×공개연도×WIPO35 분야 발명특허 수 (WIPO 커버리지 분야별 보정용).
IPC→WIPO35 대응은 _shared/ext/quality_wipo_ipc_field_concordance.csv (WIPO 2026-03판)를 읽기만 한다.
"""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, time, duckdb
OUT = _C19_OUT
EXT = os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5", r"..\..\..\_shared\ext").replace("\\", "/")
META = _C19_META
t = time.time()
con = duckdb.connect()
con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
con.sql(f"create temp table cw as select * from read_csv_auto('{EXT}/quality_wipo_ipc_field_concordance.csv', all_varchar=true)")
inv = """case when cc='KR' then substr(appl_no,1,2)='10'
 when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
 when cc='JP' then length(appl_no) in (8,17) else true end"""
q = f"""
with p as (select ksic, cc, try_cast(substr(pub_date,1,4) as int) pub_year,
   substr(main_ipc,1,4) sub, trim(split_part(substr(main_ipc,5),'/',1)) grp
 from read_parquet('{META}') where ({inv}) and pub_date is not null and pub_date<>'')
select p.ksic, p.cc, p.pub_year, coalesce(try_cast(g.wipo_field as int), try_cast(s.wipo_field as int), 0) wipo_field, count(*) n
from p left join (select * from cw where main_group is not null and main_group<>'') g on p.sub=g.subclass and p.grp=g.main_group
       left join (select * from cw where main_group is null or main_group='') s on p.sub=s.subclass
where p.pub_year between 2000 and 2025
group by all"""
d = con.sql(q).df()
d.to_parquet(os.path.join(OUT, "m1b_ksic_field_pub.parquet"), index=False)
print(len(d), "unmapped share", round(d[d.wipo_field == 0].n.sum() / d.n.sum(), 4), round(time.time() - t), "s")

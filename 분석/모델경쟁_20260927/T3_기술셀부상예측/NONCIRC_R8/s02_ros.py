# -*- coding: utf-8 -*-
"""R8 s02: RoS 셀×발행연도 표(공개 ≤t0 규칙 포함)를 R3/build_ros_pubrule_r3d.py 와 같은 집계로 만든다.
  ros_mg_issueyear_pubrule.parquet    : 셀 = 'ALL' × mg (라벨 무관, US 특허는 (mg, us_pn) 단위 1회)
  ros_pcode_issueyear_pubrule.parquet : 셀 = 제공 KSIC 의 연계표 코드 × mg (P27)
공유 작업표(r3ros_work/fam_us·rec, r3ros_patent_sci)는 읽기만 한다."""
import os, time, duckdb, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
X = os.path.join(T3, '..', '_shared', 'ext').replace(os.sep, '/')
META = os.path.join(T3, '..', '_shared', 'meta').replace(os.sep, '/') + '/*.parquet'
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
km = pd.read_csv(os.path.join(T3, 'ESSENTIAL_R7', 'conc_ksic_map.csv'), dtype=str)[['ksic', 'code']]; con.register('km', km)
con.execute(f"""create table pubrec as select cc, appl_no, ksic,
   min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
   from read_parquet('{META}') group by 1,2,3""")
con.execute(f"""create table lk as select r.ksic, km.code, r.mg, f.us_pn, r.appl_year, p.pub_year
   from read_parquet('{X}/r3ros_work/fam_us.parquet') f join read_parquet('{X}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
   left join pubrec p using(cc, appl_no, ksic) join km on km.ksic = r.ksic where r.mg is not null""")
for name, key in [('mg', "'ALL'"), ('pcode', 'code')]:
    OUT = os.path.join(HERE, f'ros_{name}_issueyear_pubrule.parquet').replace(os.sep, '/')
    if os.path.exists(OUT): continue
    con.execute(f"""copy (with cp as (select {key} ksic, mg, us_pn, min(appl_year) first_rec_year, min(pub_year) first_pub_year from lk group by 1,2,3)
       select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
       count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
       from cp c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all) to '{OUT}' (format parquet)""")
    print(name, con.sql(f"select count(*), sum(n_us_pat) from read_parquet('{OUT}')").fetchone(), round(time.time() - t), flush=True)
ref = con.sql(f"select sum(n_us_pat) from read_parquet('{T3}/R3/ros_cell_issueyear_pubrule.parquet')".replace('\\', '/')).fetchone()
print('참고: 동결 제공 셀 표 n_us_pat 합', ref)

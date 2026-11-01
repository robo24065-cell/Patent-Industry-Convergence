# -*- coding: utf-8 -*-
"""R9 s03: RoS 셀×발행연도 표(공개 ≤t0 규칙) — 셀 = TXT KSIC × mg. R7 s03(연계표 셀)과 같은 집계에서 rec_conc 대신 txt_assign 사용.
공유 작업표(_shared/ext/r3ros_work/fam_us·rec, r3ros_patent_sci)는 읽기만."""
import os, time, duckdb
# C11-T2 사본(R9 s03_ros.py, 경로만 교체). 사용: python s3b_ros.py <arm>
import sys; sys.dont_write_bytecode = True
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측"; WD = BASE + "/분석/검증루프_20261012/C11/C11-T2"; ARM = sys.argv[1]
HERE = WD + '/cache'
X = os.path.join(T3, '..', '_shared', 'ext').replace(os.sep, '/')
META = os.path.join(T3, '..', '_shared', 'meta').replace(os.sep, '/') + '/*.parquet'
TA = f'{WD}/cache/txt_assign_{ARM}.parquet'
OUT = f'{WD}/cache/ros_txt_{ARM}.parquet'
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute("set temp_directory='" + os.path.join(HERE, 'duck_tmp').replace(os.sep, '/') + "'")
con.execute(f"""create table pubrec as select cc, appl_no,
   min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
   from read_parquet('{META}') group by 1,2""")
con.execute(f"""create table cp as select ta.ksic_txt ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year, min(p.pub_year) first_pub_year
   from read_parquet('{X}/r3ros_work/fam_us.parquet') f join read_parquet('{X}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
   join read_parquet('{TA}') ta on ta.cc = f.cc and ta.appl_no = f.appl_no
   left join pubrec p on p.cc = f.cc and p.appl_no = f.appl_no
   where r.mg is not null group by 1,2,3""")
con.execute(f"""copy (select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
   count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
   from cp c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all) to '{OUT}' (format parquet)""")
print('RoS TXT 셀 표', con.sql(f"select count(*), sum(n_us_pat), count(distinct ksic) from read_parquet('{OUT}')").fetchone(), round(time.time() - t))
ref = con.sql("select sum(n_us_pat) from read_parquet('" + os.path.join(T3, 'ESSENTIAL_R7', 'ros_conc_issueyear_pubrule.parquet').replace(os.sep, '/') + "')").fetchone()
print('참고: 연계표 셀 표 n_us_pat 합', ref)

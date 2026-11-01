"""T3 3라운드 D단계 감사 수정 — RoS 셀×발행연도 표에 '배정 레코드 최초 공개연도(first_pub_year)'를 붙인다.
공유 파일 _shared/ext/r3ros_cell_issueyear.parquet 과 같은 집계(셀 = KSIC×IPC 메인그룹, US 특허 1회 배정)에
first_pub_year = 그 (셀, US 특허) 연결을 만드는 제공 레코드들의 공개연도(least(pub_date, rgst_date)) 최솟값을 추가.
build_ext_r3b.ros_feats 가 이 표를 읽어 '발행 ≤t0 & 최초 출원 ≤t0-2 & 배정 레코드 공개 ≤t0' 로 자른다.
감사(audit_ros_pub_r3d.json): 기존 규칙에서 배정 레코드가 t0 이후에만 공개된 연결이 0.25–0.48% 섞여 있었음(원점 이후 공개분 혼입) → 제거.
출력: R3/ros_cell_issueyear_pubrule.parquet (공유 폴더 파일은 수정하지 않음). 제공 데이터는 로컬에서만 읽는다.
"""
import os, time, duckdb
HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.join(HERE, '..', '..', '_shared', 'ext').replace(os.sep, '/')
META = os.path.join(HERE, '..', '..', '_shared', 'meta').replace(os.sep, '/') + '/*.parquet'
OUT = os.path.join(HERE, 'ros_cell_issueyear_pubrule.parquet').replace(os.sep, '/')
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(f"""create table pubrec as select cc, appl_no, ksic,
   min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
   from read_parquet('{META}') group by 1,2,3""")
con.execute(f"""create table cp as select r.ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year, min(p.pub_year) first_pub_year
   from read_parquet('{X}/r3ros_work/fam_us.parquet') f join read_parquet('{X}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
   left join pubrec p using(cc, appl_no, ksic) where r.mg is not null group by 1,2,3""")
con.execute(f"""copy (select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
   count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
   from cp c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all)
   to '{OUT}' (format parquet)""")
a = con.sql(f"select sum(n_us_pat), sum(n_us_pat_sci), sum(n_paper_cit), sum((first_pub_year=9999)::int) from read_parquet('{OUT}')").fetchone()
b = con.sql(f"select sum(n_us_pat), sum(n_us_pat_sci), sum(n_paper_cit) from read_parquet('{X}/r3ros_cell_issueyear.parquet')").fetchone()
print('합계 대조(새 표 / 공유 표):', a[:3], b, '| 공개연도 결측 행', a[3])
assert tuple(float(x) for x in a[:3]) == tuple(float(x) for x in b)
print('secs', round(time.time() - t))

# -*- coding: utf-8 -*-
"""Reliance on Science(pcs_oa_uspto.csv) -> 제공 데이터 연결 (family US 토큰 + US 자국 rgst_no).
입력: science_ros_raw/pcs_oa_uspto.csv, r3ros_work/fam_us.parquet, r3ros_work/rec.parquet, quality_kpss_2025.parquet(발행일 ASOF)
출력: r3ros_patent_sci.parquet   (US 특허번호 단위: 발행일 추정, 논문 인용 수 — 제공 데이터 family에 등장하는 번호만)
      r3ros_family_us_pairs.parquet (제공 레코드 cc·appl_no·ksic ↔ US 특허번호)
      r3ros_rec_sci.parquet       (제공 레코드 단위 요약)
      r3ros_cell_issueyear.parquet (KSIC×IPC메인그룹×US발행연도×최초연결출원연도: 원점 t0에서 issue_year≤t0 & first_rec_year≤t0 로 자르면 시점정합)
      r3ros_ksic_issueyear.parquet (50 KSIC × 발행연도)
      r3ros_link_rate_cc_year.csv, r3ros_link_by_ksic.csv
"""
import os, time, duckdb
from pathlib import Path
EXT = Path(__file__).resolve().parent
E = EXT.as_posix()
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3500MB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(f"set temp_directory='{E}/r3ros_work/tmp'")
CSV = f"{E}/science_ros_raw/pcs_oa_uspto.csv"
if not (EXT / "r3ros_work" / "cit.parquet").exists():
    con.execute(f"""copy (select reftype, confscore::tinyint confscore, oaid::bigint oaid,
          try_cast(regexp_extract(patent, '^us-([0-9]+)-', 1) as bigint) pn, wherefound
        from read_csv('{CSV}', header=true, columns={{'reftype':'VARCHAR','confscore':'INTEGER','oaid':'VARCHAR','patent':'VARCHAR','wherefound':'VARCHAR'}}))
        to '{E}/r3ros_work/cit.parquet' (format parquet)""")
    print('cit.parquet', round(time.time() - t), flush=True)
C = f"read_parquet('{E}/r3ros_work/cit.parquet')"
print(con.sql(f"select count(*) n, count(distinct pn) npat, count(distinct oaid) nwork, sum((pn is null)::int) pn_null from {C}").df().to_string())
print(con.sql(f"select confscore, count(*) n from {C} group by 1 order by 1").df().T.to_string())
print(con.sql(f"select reftype, wherefound, count(*) n from {C} group by 1,2 order by 1,2").df().to_string())
# 발행일: KPSS(patent_num→issue_date) ASOF (번호는 발행일 순으로 단조 증가)
con.execute(f"create table kp as select patent_num, issue_date from read_parquet('{E}/quality_kpss_2025.parquet') where patent_num is not null")
con.execute(f"create table fam as select * from read_parquet('{E}/r3ros_work/fam_us.parquet') where us_pn between 3000000 and 13000000")
con.execute("create table upn as select distinct us_pn pn from fam")
con.execute("create table upn_d as select u.pn, k.issue_date from upn u asof left join kp k on u.pn >= k.patent_num")
# 논문 인용(신뢰도 ≥3 기본; ≥1·≥5 병기), 앞면/본문
con.execute(f"""create table ps as select pn,
   count(*) filter (where confscore>=1) n_cit_c1,
   count(*) filter (where confscore>=3) n_cit,
   count(*) filter (where confscore>=5) n_cit_c5,
   count(*) filter (where confscore>=3 and wherefound in ('frontonly','both')) n_cit_front,
   count(*) filter (where confscore>=3 and wherefound='bodyonly') n_cit_body,
   count(*) filter (where confscore>=3 and reftype='exm') n_cit_exm
  from {C} where pn in (select pn from upn) group by pn""")
con.execute(f"""copy (select d.pn us_pn, d.issue_date, year(d.issue_date) issue_year,
     coalesce(n_cit_c1,0) n_cit_c1, coalesce(n_cit,0) n_cit, coalesce(n_cit_c5,0) n_cit_c5,
     coalesce(n_cit_front,0) n_cit_front, coalesce(n_cit_body,0) n_cit_body, coalesce(n_cit_exm,0) n_cit_exm
   from upn_d d left join ps using(pn)) to '{E}/r3ros_patent_sci.parquet' (format parquet)""")
con.execute(f"create table P as select * from read_parquet('{E}/r3ros_patent_sci.parquet')")
print('patent_sci', con.sql("select count(*), sum((n_cit>0)::int), avg((n_cit>0)::int), sum((issue_date is null)::int) from P").fetchall(), round(time.time() - t), flush=True)
print(con.sql("select issue_year, count(*) n_us, round(avg((n_cit>0)::int),4) sci_share, round(avg((n_cit_front>0)::int),4) front_share, round(avg(n_cit),2) mean_cit from P where issue_year between 1995 and 2025 group by 1 order by 1").df().to_string())
con.execute(f"copy (select cc, appl_no, ksic, us_pn, src from fam) to '{E}/r3ros_family_us_pairs.parquet' (format parquet)")
con.execute(f"create table rec as select * from read_parquet('{E}/r3ros_work/rec.parquet')")
con.execute("""create table rs as select f.cc, f.appl_no, f.ksic, count(*) n_us, sum((p.n_cit>0)::int) n_us_sci,
     sum(p.n_cit) n_paper_cit, sum(p.n_cit_front) n_paper_cit_front, min(p.issue_year) first_issue_year,
     max((f.src='own')::int) own_linked
   from fam f join P p on f.us_pn=p.us_pn group by 1,2,3""")
con.execute(f"""copy (select r.cc, r.appl_no, r.ksic, r.appl_year, r.mg, coalesce(n_us,0) n_us, coalesce(n_us_sci,0) n_us_sci,
     coalesce(n_paper_cit,0) n_paper_cit, coalesce(n_paper_cit_front,0) n_paper_cit_front, first_issue_year, coalesce(own_linked,0) own_linked
   from rec r left join rs using(cc, appl_no, ksic)) to '{E}/r3ros_rec_sci.parquet' (format parquet)""")
R = f"read_parquet('{E}/r3ros_rec_sci.parquet')"
lr = con.sql(f"""select cc, appl_year, count(*) n_rec, avg((n_us>0)::int) link_rate, avg((n_us_sci>0)::int) sci_rate_all,
     sum((n_us_sci>0)::int)/nullif(sum((n_us>0)::int),0) sci_rate_linked
   from {R} group by 1,2 order by 1,2""").df()
lr.round(4).to_csv(EXT / "r3ros_link_rate_cc_year.csv", index=False, encoding="utf-8-sig")
print(lr[(lr.appl_year >= 2000) & (lr.appl_year <= 2023)].pivot(index='appl_year', columns='cc', values='link_rate').round(3).to_string())
lk = con.sql(f"""select ksic, count(*) n_rec, avg((n_us>0)::int) link_rate, sum((n_us>0)::int) n_linked,
     sum((n_us>0 and appl_year<=2016)::int) n_linked_le2016,
     sum((n_us_sci>0)::int)/nullif(sum((n_us>0)::int),0) sci_rate_linked,
     avg((n_us>0)::int) filter (where cc='KR') link_rate_kr, avg((n_us>0)::int) filter (where cc='US') link_rate_us
   from {R} group by 1 order by 1""").df()
lk.round(4).to_csv(EXT / "r3ros_link_by_ksic.csv", index=False, encoding="utf-8-sig")
print('KSIC', len(lk), 'link min/med/max', lk.link_rate.min().round(3), lk.link_rate.median().round(3), lk.link_rate.max().round(3),
      '| sci_rate_linked min/med/max', lk.sci_rate_linked.min().round(3), lk.sci_rate_linked.median().round(3), lk.sci_rate_linked.max().round(3))
# 셀 × US 발행연도 패널 (US 특허번호를 셀에 1회만 배정: 같은 셀의 여러 레코드가 같은 번호를 가리켜도 1)
con.execute("""create table cp as select r.ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year
   from fam f join rec r using(cc, appl_no, ksic) where r.mg is not null group by 1,2,3""")
con.execute(f"""copy (select c.ksic, c.mg, p.issue_year, c.first_rec_year, count(*) n_us_pat, sum((p.n_cit>0)::int) n_us_pat_sci,
     sum(p.n_cit) n_paper_cit, sum(p.n_cit_front) n_paper_cit_front, sum((p.n_cit_front>0)::int) n_us_pat_sci_front
   from cp c join P p using(us_pn) where p.issue_year is not null group by all) to '{E}/r3ros_cell_issueyear.parquet' (format parquet)""")
con.execute(f"""copy (select r.ksic, p.issue_year, count(distinct p.us_pn) n_us_pat, count(distinct p.us_pn) filter (where p.n_cit>0) n_us_pat_sci,
     sum(p.n_cit) n_paper_cit
   from (select distinct ksic, us_pn from fam) r join P p using(us_pn) where p.issue_year is not null group by all) to '{E}/r3ros_ksic_issueyear.parquet' (format parquet)""")
print(con.sql(f"select count(*), count(distinct ksic||mg), sum(n_us_pat) from read_parquet('{E}/r3ros_cell_issueyear.parquet')").fetchall())
print('done', round(time.time() - t))

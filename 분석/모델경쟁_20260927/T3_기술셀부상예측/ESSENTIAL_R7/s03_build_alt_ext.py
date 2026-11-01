# -*- coding: utf-8 -*-
"""R7 s03: 연계표 셀(코드×IPC 메인그룹)용 외부 피처.
  (1) KOSIS 광공업생산지수 성장률 3피처: 동결 산출물(R3/ext_ksic_t0_pub.parquet, R2/ext_ksic_t0.parquet)의 KSIC별 값을
      연계표 코드별로 평균(같은 코드의 KSIC 는 대부분 같은 KOSIS 3자리 계열) → ext_pi_conc_{pub,app}.parquet (ksic=코드)
  (2) RoS 셀 표: R3/build_ros_pubrule_r3d.py 와 같은 집계를 셀 = 연계표 코드×메인그룹으로 → ros_conc_issueyear_pubrule.parquet
      (공유 RoS 작업표 r3ros_work/fam_us·rec, r3ros_patent_sci 는 읽기만)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE)
import numpy as np, pandas as pd, duckdb
X = os.path.join(T3, '..', '_shared', 'ext').replace(os.sep, '/')
META = os.path.join(T3, '..', '_shared', 'meta').replace(os.sep, '/') + '/*.parquet'
km = pd.read_csv(os.path.join(HERE, 'conc_ksic_map.csv'), dtype=str)[['ksic', 'code']]
PI = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']
for var, src in [('pub', os.path.join(T3, 'R3', 'ext_ksic_t0_pub.parquet')), ('app', os.path.join(T3, 'R2', 'ext_ksic_t0.parquet')),
                 ('pubF', os.path.join(T3, 'R3', 'ext_ksic_t0_pub_final.parquet'))]:
    e = pd.read_parquet(src)[['ksic', 't0'] + PI].merge(km, on='ksic')
    sd = e.groupby(['code', 't0'])[PI].std().max().max()
    o = e.groupby(['code', 't0'])[PI].mean().reset_index().rename(columns={'code': 'ksic'})
    o.to_parquet(os.path.join(HERE, f'ext_pi_conc_{var}.parquet'))
    print(var, o.shape, '같은 코드 안 KSIC 간 최대 표준편차', round(float(sd), 4))
t = time.time()
for OUT, RCF in [(os.path.join(HERE, 'ros_conc_issueyear_pubrule.parquet').replace(os.sep, '/'), 'rec_conc.parquet'),
                 (os.path.join(HERE, 'ros_concm_issueyear_pubrule.parquet').replace(os.sep, '/'), 'rec_conc_multi.parquet')]:
  if not os.path.exists(OUT):
      con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
      c50 = "(" + ",".join(f"'{c}'" for c in sorted(km.code.unique())) + ")"
      RC = os.path.join(HERE, RCF).replace(os.sep, '/')
      con.execute(f"""create table pubrec as select cc, appl_no,
         min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
         from read_parquet('{META}') group by 1,2""")
      con.execute(f"""create table cp as select rc.code ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year, min(p.pub_year) first_pub_year
         from read_parquet('{X}/r3ros_work/fam_us.parquet') f join read_parquet('{X}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
         join read_parquet('{RC}') rc on rc.cc = f.cc and rc.appl_no = f.appl_no
         left join pubrec p on p.cc = f.cc and p.appl_no = f.appl_no
         where r.mg is not null and rc.code in {c50} group by 1,2,3""")
      con.execute(f"""copy (select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
         count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
         from cp c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all)
         to '{OUT}' (format parquet)""")
      print('RoS 연계표 셀 표', con.sql(f"select count(*), sum(n_us_pat) from read_parquet('{OUT}')").fetchone(), round(time.time() - t))

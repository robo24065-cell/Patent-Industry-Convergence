# C02-T2 s1: 제공 KR 등록 연도·월·코호트 집계 + 등록원부 분해 (읽기 전용)
import os, json, time, sqlite3, sys
import duckdb, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
REG = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\cache")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
con.sql(f"""create temp table kr as select distinct appl_no, appl_date, appl_year, rgst_no, rgst_date, pub_date, ksic, prio_cc, applicant
            from read_parquet('{META}') where cc='KR'""")
res = {}
res['kr_rows'] = con.sql("select count(*), count(distinct appl_no), count(rgst_no), count(distinct rgst_no) from kr").fetchall()
res['rgst_prefix'] = con.sql("select substr(rgst_no,1,2) p, count(distinct rgst_no) from kr where rgst_no is not null group by 1 order by 1").fetchall()
res['appl_prefix'] = con.sql("select substr(appl_no,1,2) p, count(distinct appl_no), count(distinct rgst_no) from kr group by 1 order by 1").fetchall()
y = con.sql("""select substr(rgst_date,1,4)::int y, substr(rgst_no,1,2) p, count(distinct rgst_no) n from kr where rgst_no is not null group by 1,2 order by 1,2""").df()
y.to_csv(os.path.join(OUT, 's1_kr_rgst_year.csv'), index=False)
m = con.sql("""select substr(rgst_date,1,6) ym, count(distinct rgst_no) n,
                 count(distinct rgst_no) filter (where appl_year<=2019) n_old, count(distinct rgst_no) filter (where appl_year>=2020) n_new
               from kr where rgst_no is not null and substr(rgst_no,1,2)='10' and rgst_date>='20210101' group by 1 order by 1""").df()
m.to_csv(os.path.join(OUT, 's1_kr_rgst_month.csv'), index=False)
# 등록연도 × 경과(등록연도−출원연도)
a = con.sql("""select substr(rgst_date,1,4)::int y, (substr(rgst_date,1,4)::int - appl_year) age, count(distinct rgst_no) n
               from kr where rgst_no is not null and substr(rgst_no,1,2)='10' group by 1,2""").df()
a.to_csv(os.path.join(OUT, 's1_kr_rgst_year_age.csv'), index=False)
# 출원연도 코호트별 등록률(출원연도별 출원 수, 등록 수, 등록연도 분포)
c = con.sql("""select appl_year, count(distinct appl_no) n_app, count(distinct rgst_no) n_reg,
                 count(distinct rgst_no) filter (where rgst_date<'20240101') reg_by2023,
                 count(distinct rgst_no) filter (where rgst_date between '20240101' and '20241231') reg_2024,
                 count(distinct rgst_no) filter (where rgst_date>='20250101') reg_2025
               from kr where substr(appl_no,1,2)='10' group by 1 order by 1""").df()
c.to_csv(os.path.join(OUT, 's1_kr_cohort.csv'), index=False)
# KSIC × 등록연도 2022–2024, 해외 우선권
k = con.sql("""select ksic, substr(rgst_date,1,4)::int y, count(distinct rgst_no) n,
                 count(distinct rgst_no) filter (where prio_cc is not null and regexp_replace(replace(prio_cc,'KR',''),'\\|','','g')<>'') n_prio_for
               from kr where rgst_no is not null and substr(rgst_no,1,2)='10' and rgst_date between '20190101' and '20251231' group by 1,2""").df()
k.to_csv(os.path.join(OUT, 's1_kr_ksic_year.csv'), index=False)
con.close()
# 등록원부: 출원인 국적·유형(대표 출원인 = seq 최소)
rc = sqlite3.connect(f'file:{REG}?mode=ro', uri=True)
r = pd.read_sql("""select p.rgst_no, substr(p.rgstDate,1,4) y, substr(p.rgstDate,1,6) ym, substr(p.applDate,1,4) ay, p.ksic, p.applTpcd,
                          a.applicantCd, a.applicantNatl
                   from patent p left join (select rgst_no, applicantCd, applicantNatl, min(seq) s from applicant group by rgst_no) a on a.rgst_no=p.rgst_no""", rc)
rc.close()
r['natl'] = (r.applicantNatl == '대한민국').map({True: '내국', False: '외국'})
r.loc[r.applicantNatl.isna(), 'natl'] = '결측'
r['atype'] = r.applicantCd.str[:1].fillna('결측')
r.to_parquet(os.path.join(OUT, 's1_registry_slim.parquet'), index=False)
res['registry_year'] = r.groupby('y').size().loc['2019':].to_dict()
res['registry_natl_year'] = r[r.y.between('2019', '2025')].groupby(['y', 'natl']).size().unstack().to_dict()
res['registry_atype_year'] = r[r.y.between('2019', '2025')].groupby(['y', 'atype']).size().unstack().to_dict()
res['registry_tpcd_year'] = r[r.y.between('2019', '2025')].groupby(['y', 'applTpcd']).size().unstack().to_dict()
json.dump(res, open(os.path.join(OUT, 's1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str)[:3000]); print(y[y.y >= 2012].to_string()); print(m.to_string()); print(round(time.time() - t, 1))

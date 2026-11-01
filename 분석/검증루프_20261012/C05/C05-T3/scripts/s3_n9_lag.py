# -*- coding: utf-8 -*-
"""C05-T3 s3: N9 최종창 보고 완결도(절단) 원천 재계산.
(1) 제공 KR 출원연도별 NTIS 연결률(전체·수행주체별), 2014–2021 평균 대비 상대값
(2) 과제연도 t별 lag0(출원=t)·lag1(출원=t+1) 연결 특허 수와 비 lag1/lag0, 과제당 양성률
(3) KR 출원 월별 건수(12월 결손) — 2023-12 대 이전 해 12월
출력 cache/s3_out.json, cache/s3_linkrate.csv, cache/s3_lagratio.csv
"""
import duckdb, json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
SH = os.path.join(B, r"모델경쟁_20260927\_shared")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
c = duckdb.connect(); c.execute("SET memory_limit='3GB'; SET threads=2")
M = os.path.join(SH, "meta", "*.parquet").replace("\\", "/")
P = os.path.join(SH, "ext", "ntis_project_raw.parquet").replace("\\", "/")
LA = os.path.join(CA, "n9_links_A.parquet").replace("\\", "/")
R = os.path.join(SH, "ext", "ntis_rpatent_raw.parquet").replace("\\", "/")
o = {}
c.execute(f"create table kr as select appl_no, any_value(appl_year)::int ay, any_value(appl_date) ad from read_parquet('{M}') where cc='KR' group by 1")
c.execute(f"create table la as select * from read_parquet('{LA}')")
# performer group of each link from raw (PerformAgent)
c.execute(f"create table pa as select ResultID, PerformAgent from read_parquet('{R}')")
c.execute("""create table lap as select la.*, case when PerformAgent like '%대학%' then '대학' when PerformAgent like '%출연%' then '출연연'
   when PerformAgent like '%기업%' then '기업' when PerformAgent like '%국공립%' then '국공립연' else '기타' end grp from la join pa using(ResultID)""")
q = """with lk as (select appl_no, max((grp='대학')::int) u, max((grp='출연연')::int) g, max((grp='기업')::int) f from lap group by 1)
 select ay, count(*) n, count(lk.appl_no) nl, count(lk.appl_no)/count(*) rate, sum(coalesce(u,0))/count(*) r_univ,
   sum(coalesce(g,0))/count(*) r_gri, sum(coalesce(f,0))/count(*) r_firm
 from kr left join lk using(appl_no) where ay between 2007 and 2025 group by 1 order by 1"""
df = c.sql(q).df()
base = df[(df.ay >= 2014) & (df.ay <= 2021)][['rate', 'r_univ', 'r_gri', 'r_firm']].mean()
for col in ['rate', 'r_univ', 'r_gri', 'r_firm']:
    df[col + '_rel1421'] = df[col] / base[col]
base2 = df[(df.ay >= 2014) & (df.ay <= 2016)][['rate']].mean()
df['rate_rel1416'] = df['rate'] / base2['rate']
df.to_csv(os.path.join(CA, 's3_linkrate.csv'), index=False, encoding='utf-8-sig')
o['linkrate'] = df.round(4).to_dict('records')
# (2) lag ratio by project year t: links whose ProjectID is a project of year t (pyear from project raw)
c.execute(f"create table pj as select pjt_no, year::int t, perform_grp from read_parquet('{P}') where not coalesce(secret_flag,false)")
c.execute("create table lp as select distinct pj.t, pj.perform_grp, la.ProjectID, la.appl_no, la.appl_year::int ay from la join pj on la.ProjectID=pj.pjt_no")
lag = c.sql("""select t, count(distinct case when ay=t then appl_no||ProjectID end) lag0, count(distinct case when ay=t+1 then appl_no||ProjectID end) lag1,
   count(distinct case when ay=t-1 then appl_no||ProjectID end) lagm1
   from lp where t between 2010 and 2024 group by 1 order by 1""").df()
lag['ratio_lag1_lag0'] = lag.lag1 / lag.lag0
npj = c.sql("""select t, count(*) n_proj from pj where t between 2010 and 2024 group by 1""").df()
pos = c.sql("""select t, count(distinct ProjectID) n_pos_window from lp where ay between t and t+1 group by 1""").df()
lag = lag.merge(npj, on='t').merge(pos, on='t', how='left')
lag['pos_rate_window'] = lag.n_pos_window / lag.n_proj
lag['pos_rate_lag0only'] = None
p0 = c.sql("select t, count(distinct ProjectID) n0 from lp where ay=t group by 1").df()
lag = lag.merge(p0, on='t', how='left'); lag['pos_rate_lag0only'] = lag.n0 / lag.n_proj
# by performer group for lag ratio
lagg = c.sql("""select t, case when perform_grp in ('대기업','중소기업','중견기업') then '기업' when perform_grp in ('출연연','국공립연') then '출연연·국공립연'
   when perform_grp='대학' then '대학' else '기타' end g,
   count(distinct case when ay=t then appl_no||ProjectID end) lag0, count(distinct case when ay=t+1 then appl_no||ProjectID end) lag1
   from lp where t between 2014 and 2023 group by 1,2 order by 2,1""").df()
lagg['ratio'] = lagg.lag1 / lagg.lag0
lag.to_csv(os.path.join(CA, 's3_lagratio.csv'), index=False, encoding='utf-8-sig')
lagg.to_csv(os.path.join(CA, 's3_lagratio_grp.csv'), index=False, encoding='utf-8-sig')
o['lag'] = lag.round(4).to_dict('records')
o['lag_grp'] = lagg.round(4).to_dict('records')
# (3) monthly KR application counts
mon = c.sql("""select ay, substr(ad,5,2) mm, count(*) n from kr where ay between 2015 and 2024 group by 1,2 order by 1,2""").df()
pv = mon.pivot(index='ay', columns='mm', values='n')
pv['dec_over_janNov_mean'] = pv['12'] / pv[[f'{m:02d}' for m in range(1, 12)]].mean(axis=1)
o['dec_ratio'] = pv['dec_over_janNov_mean'].round(4).to_dict()
o['dec_ratio_2016_2022_mean'] = float(pv.loc[2016:2022, 'dec_over_janNov_mean'].mean())
o['kr_appl_count_by_year'] = pv[[f'{m:02d}' for m in range(1, 13)]].sum(axis=1).astype(int).to_dict()
json.dump(o, open(os.path.join(CA, 's3_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(df[['ay', 'n', 'nl', 'rate', 'rate_rel1421', 'rate_rel1416', 'r_univ_rel1421', 'r_gri_rel1421', 'r_firm_rel1421']].round(3).to_string())
print(lag.round(3).to_string())
print(lagg.round(3).to_string())
print(o['dec_ratio'], o['dec_ratio_2016_2022_mean'])

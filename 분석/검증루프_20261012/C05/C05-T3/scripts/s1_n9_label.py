# -*- coding: utf-8 -*-
"""C05-T3 s1: N9 최종시험(t=2022) 레이블을 원천에서 독립 재구성 (트랙 코드 미사용).
R-A: ntis_rpatent_raw(KR 13자리) -> 제공 meta KR(appl_no|rgst_no) -> appl_year in [2022,2023]
R-B: 공용 ntis_patent_link (ProjectID, appl_year)
출력: cache/n9_units.parquet, cache/n9_links_A.parquet, n9_label_rebuild.csv, cache/s1_out.json
"""
import duckdb, json, os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N9 = os.path.join(B, r"모델경쟁_20260927\N9_정부RnD과제_특허성과예측")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3")
CA = os.path.join(OUT, "cache"); os.makedirs(CA, exist_ok=True)
t0 = time.time()
c = duckdb.connect()
c.execute("SET memory_limit='3GB'; SET threads=2")
P = os.path.join(SH, "ext", "ntis_project_raw.parquet").replace("\\", "/")
R = os.path.join(SH, "ext", "ntis_rpatent_raw.parquet").replace("\\", "/")
L = os.path.join(SH, "ext", "ntis_patent_link.parquet").replace("\\", "/")
M = os.path.join(SH, "meta", "*.parquet").replace("\\", "/")
PR = os.path.join(N9, "final_preds_r6.parquet").replace("\\", "/")
o = {}
# --- units
o['secret_flag_dist_2022'] = c.sql(f"select cast(secret_flag as varchar) s, count(*) n from read_parquet('{P}') where cast(year as int)=2022 group by 1 order by 2 desc").fetchall()
c.execute(f"""create table units as
 select pjt_no, count(*) nrow, any_value(perform_grp) perform_grp, any_value(research_agency) agency
 from read_parquet('{P}') where cast(year as int)=2022 and not coalesce(secret_flag,false)
 group by pjt_no""")
o['units_n'] = c.sql("select count(*) from units").fetchone()[0]
o['units_dup_pjt'] = c.sql("select count(*) from units where nrow>1").fetchone()[0]
# --- KR provided patents (one row per appl_no)
c.execute(f"""create table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(appl_year) appl_year,
  any_value(appl_date) appl_date, any_value(pub_date) pub_date, any_value(rgst_date) rgst_date, any_value(ksic) ksic, count(*) nks
  from read_parquet('{M}') where cc='KR' group by appl_no""")
o['kr_n'] = c.sql("select count(*) from kr").fetchone()[0]
o['kr_rgst_dup'] = c.sql("select count(*) from (select rgst_no from kr where rgst_no is not null group by 1 having count(*)>1)").fetchone()[0]
# --- NTIS raw KR results
c.execute(f"""create table rp as select ResultID, ProjectID, cast(ProjectYear as int) pyear, cast(Year as int) ryear,
   RegistType, IprType, num_digits num, PerformAgent from read_parquet('{R}') where is_kr and length(num_digits)=13""")
o['rp_kr13'] = c.sql("select count(*) from rp").fetchone()[0]
o['rp_dup_resultid'] = c.sql("select count(*)-count(distinct ResultID) from rp").fetchone()[0]
o['rp_dup_full'] = c.sql("select count(*) from (select ResultID, ProjectID, num from rp group by all having count(*)>1)").fetchone()[0]
c.execute("""create table links_A as
  select distinct rp.ResultID, rp.ProjectID, rp.pyear, rp.ryear, rp.RegistType, rp.IprType, kr.appl_no, kr.appl_year, kr.ksic, 'appl' route
  from rp join kr on rp.num = kr.appl_no
  union
  select distinct rp.ResultID, rp.ProjectID, rp.pyear, rp.ryear, rp.RegistType, rp.IprType, kr.appl_no, kr.appl_year, kr.ksic, 'rgst' route
  from rp join kr on rp.num = kr.rgst_no""")
o['links_A_rows'] = c.sql("select count(*) from links_A").fetchone()[0]
o['links_A_patents'] = c.sql("select count(distinct appl_no) from links_A").fetchone()[0]
o['links_A_route'] = c.sql("select route, count(*) from links_A group by 1").fetchall()
c.execute(f"copy links_A to '{os.path.join(CA,'n9_links_A.parquet')}' (format parquet)")
# --- R-B shared link
c.execute(f"create table links_B as select distinct ProjectID, appl_no, cast(appl_year as int) appl_year, ResultID, match_route from read_parquet('{L}')")
o['links_B_rows'] = c.sql("select count(*) from links_B").fetchone()[0]
o['links_B_patents'] = c.sql("select count(distinct appl_no) from links_B").fetchone()[0]
# compare link sets (patent x project)
o['AB_pairs'] = c.sql("""select
  (select count(*) from (select distinct ProjectID, appl_no from links_A)) a,
  (select count(*) from (select distinct ProjectID, appl_no from links_B)) b,
  (select count(*) from (select distinct ProjectID, appl_no from links_A) x join (select distinct ProjectID, appl_no from links_B) y using(ProjectID, appl_no)) ab""").fetchone()
# --- labels
def lab(tbl, name, extra=""):
    c.execute(f"""create or replace table {name} as select ProjectID pjt_no, count(distinct appl_no) n_pat, count(distinct ResultID) n_res
      from {tbl} where appl_year between 2022 and 2023 {extra} group by 1""")
lab('links_A', 'yA'); lab('links_B', 'yB')
lab('links_A', 'yA_pat', "and IprType like '%특허%'")
lab('links_A', 'yA_pyear', "and pyear=2022")
lab('links_A', 'yA_ryear', "and ryear between 2022 and 2023 and RegistType like '%출원%'")
lab('links_A', 'yA_noreg', "and route='appl'")
c.execute(f"create table pr as select pjt_no, y y_off, org from read_parquet('{PR}')")
c.execute("""create table lab as select coalesce(u.pjt_no, pr.pjt_no) pjt_no, u.pjt_no is not null in_units, pr.pjt_no is not null in_preds,
   pr.y_off, u.perform_grp, u.agency,
   coalesce(a.n_pat,0) nA, coalesce(b.n_pat,0) nB,
   (a.pjt_no is not null)::int yA, (b.pjt_no is not null)::int yB,
   (ap.pjt_no is not null)::int yA_pat, (ay.pjt_no is not null)::int yA_pyear,
   (ar.pjt_no is not null)::int yA_ryear, (an.pjt_no is not null)::int yA_noreg
   from units u full outer join pr using(pjt_no)
   left join yA a on a.pjt_no=coalesce(u.pjt_no, pr.pjt_no)
   left join yB b on b.pjt_no=coalesce(u.pjt_no, pr.pjt_no)
   left join yA_pat ap on ap.pjt_no=coalesce(u.pjt_no, pr.pjt_no)
   left join yA_pyear ay on ay.pjt_no=coalesce(u.pjt_no, pr.pjt_no)
   left join yA_ryear ar on ar.pjt_no=coalesce(u.pjt_no, pr.pjt_no)
   left join yA_noreg an on an.pjt_no=coalesce(u.pjt_no, pr.pjt_no)""")
o['unit_match'] = c.sql("select sum(in_units::int), sum(in_preds::int), sum((in_units and in_preds)::int), count(*) from lab").fetchone()
o['pos'] = c.sql("""select sum(y_off) official, sum(case when in_preds then yA end) A, sum(case when in_preds then yB end) B,
   sum(case when in_preds then yA_pat end) A_pat, sum(case when in_preds then yA_pyear end) A_pyear,
   sum(case when in_preds then yA_ryear end) A_ryear, sum(case when in_preds then yA_noreg end) A_noreg,
   sum(case when in_units then yA end) A_allunits from lab""").fetchone()
o['agree'] = c.sql("""select avg((yA=y_off)::int), avg((yB=y_off)::int), sum((yA=1 and y_off=0)::int), sum((yA=0 and y_off=1)::int),
   sum((yB=1 and y_off=0)::int), sum((yB=0 and y_off=1)::int) from lab where in_preds""").fetchone()
# positive project linked patents beyond preds units (project-id appearing in rpatent with 2022 but not in units)
o['yA_pos_projects_not_in_units'] = c.sql("select count(*) from yA where pjt_no not in (select pjt_no from units)").fetchone()[0]
o['yA_pos_projects_not_in_units_secret'] = c.sql(f"""select count(*) from yA where pjt_no in (select pjt_no from read_parquet('{P}') where cast(year as int)=2022
     and coalesce(secret_flag,false))""").fetchone()[0]
# multi-project patents in final window
o['final_window_patents_A'] = c.sql("select count(distinct appl_no) from links_A where appl_year between 2022 and 2023").fetchone()[0]
o['final_window_patents_multi_project'] = c.sql("""select count(*) from (select appl_no from links_A where appl_year between 2022 and 2023
     group by 1 having count(distinct ProjectID)>1)""").fetchone()[0]
o['final_window_dup_records_same_patent_project'] = c.sql("""select count(*) from (select appl_no, ProjectID from links_A where appl_year between 2022 and 2023
     group by 1,2 having count(distinct ResultID)>1)""").fetchone()[0]
c.execute(f"copy (select * from lab order by pjt_no) to '{os.path.join(OUT,'n9_label_rebuild.csv')}' (header, delimiter ',')")
c.execute(f"copy lab to '{os.path.join(CA,'n9_lab.parquet')}' (format parquet)")
o['sec'] = round(time.time() - t0, 1)
json.dump(o, open(os.path.join(CA, 's1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))

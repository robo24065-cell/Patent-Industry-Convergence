"""T3 2라운드 추가 셀 패널: 출원인 유형(대학·공공연구기관·개인)과 KR 등록 속도.
산출: cache_dev/feat_r2_panel.parquet (ksic, mg, cc, year, n_univ, n_gri, n_corp_kw), cache_dev/kr_reg_panel.parquet (ksic, mg, appl_year, rgst_year, n)
🔒 출원연도 ≤ MAXYEAR(2019), 등록연도도 ≤ MAXYEAR 만 남김(그 이후 등록은 NULL = '아직 미등록'으로 취급).
"""
import os, sys, time, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MAXY = int(os.environ.get('T3R2_MAXYEAR', '2019'))
if MAXY >= 2020 and os.environ.get('T3R2_FINAL_UNLOCK') != '1': raise RuntimeError('🔒')
CACHE = os.path.join(HERE, os.environ.get('T3R2_CACHE', 'cache_dev'))
META = os.path.join(HERE, '..', '..', '_shared', 'meta', '*.parquet').replace('\\', '/')
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
src = open(os.path.join(HERE, '..', 'build_panel.py'), encoding='utf-8').read()
exec(src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')])
B = os.path.join(CACHE, 'base_rows.parquet').replace('\\', '/')
con.execute(rf"""create temp table a as select ksic, mgn(ipc_raw) mg, cc, year, appl_no, any_value(app1) app1
  from read_parquet('{B}') where mgn(ipc_raw) is not null and year <= {MAXY} group by all""")
UNIV = r"(대학|학교|산학협력|university|univ\.|college|institute of technology|大学|大學|学院|학원)"
GRI = r"(연구원|연구소|연구재단|한국과학기술|research institute|national lab|academy of sciences|科学院|研究院|研究所|fraunhofer|cnrs|commissariat|agency)"
CORP = r"(주식회사|\(주\)|㈜|co\.|ltd|inc|corp|gmbh|s\.a\.|ag$|llc|company|公司|株式会社|有限)"
con.execute(rf"""copy (select ksic, mg, cc, year,
    count(*) filter (where regexp_matches(lower(app1), '{UNIV}')) n_univ,
    count(*) filter (where regexp_matches(lower(app1), '{GRI}') and not regexp_matches(lower(app1), '{UNIV}')) n_gri,
    count(*) filter (where cc='KR' and regexp_matches(trim(app1), '^[가-힣]{{2,4}}$')) n_indiv   -- KR 개인(한글 2~4자 성명). 타국은 US 구법상 발명자=출원인이라 제외
  from a group by all) to '{CACHE}/feat_r2_panel.parquet' (format parquet)""".replace('\\', '/'))
print('feat_r2_panel', time.time() - t, flush=True)
con.execute(rf"""copy (
  select ksic, mgn(regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g')) mg, appl_year::int appl_year,
     case when rgst_date is not null and rgst_date<>'' and cast(substr(rgst_date,1,4) as int) <= {MAXY} then cast(substr(rgst_date,1,4) as int) end rgst_year,
     count(distinct appl_no) n
  from read_parquet('{META}') where cc='KR' and appl_year between 2000 and {MAXY} group by all
) to '{CACHE}/kr_reg_panel.parquet' (format parquet)""")
print(con.sql(f"select count(*), sum(n), sum(n) filter (where rgst_year is not null) from read_parquet('{CACHE}/kr_reg_panel.parquet')".replace('\\', '/')).fetchall())
print(con.sql(f"select cc, sum(n_univ), sum(n_gri), sum(n_indiv) from read_parquet('{CACHE}/feat_r2_panel.parquet') group by cc".replace('\\', '/')).df())
print('done', time.time() - t)

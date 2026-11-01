# -*- coding: utf-8 -*-
"""제공 메타에서 R8 파일럿에 필요한 최소 컬럼만 뽑아 캐시. 원본은 읽기만."""
import sys
sys.dont_write_bytecode = True
from r8_common import *
import duckdb

con = duckdb.connect()
con.sql("SET threads=2; SET memory_limit='3GB'; SET preserve_insertion_order=false")
recp = p("cache", "rec.parquet").replace("\\", "/")
if not os.path.exists(recp):
    con.sql(f"""
    COPY (
      WITH b AS (
        SELECT cc, appl_no, appl_year::INT AS appl_year, ksic,
          regexp_replace(coalesce(main_ipc,''), '\\s', '', 'g') AS ipc,
          try_strptime(pub_date, '%Y%m%d') AS pd, try_strptime(rgst_date, '%Y%m%d') AS rd
        FROM read_parquet('{META}'))
      SELECT cc, appl_no, appl_year, ksic,
        coalesce(least(year(pd), year(rd)), year(pd), year(rd), appl_year + 2)::INT AS pyear,
        regexp_extract(ipc, '^([A-H][0-9]{{2}}[A-Z])', 1) AS sc,
        CASE WHEN regexp_matches(ipc, '^[A-H][0-9]{{2}}[A-Z][0-9]+/') THEN
          regexp_extract(ipc, '^([A-H][0-9]{{2}}[A-Z])', 1) || CAST(CAST(regexp_extract(ipc, '^[A-H][0-9]{{2}}[A-Z]([0-9]+)/', 1) AS INT) AS VARCHAR)
        ELSE '' END AS mg,
        CASE WHEN regexp_matches(ipc, '^[A-H][0-9]{{2}}[A-Z][0-9]+/[0-9]+') THEN
          regexp_extract(ipc, '^([A-H][0-9]{{2}}[A-Z])', 1) || CAST(CAST(regexp_extract(ipc, '^[A-H][0-9]{{2}}[A-Z]([0-9]+)/', 1) AS INT) AS VARCHAR)
          || '/' || regexp_extract(ipc, '^[A-H][0-9]{{2}}[A-Z][0-9]+/([0-9]+)', 1)
        ELSE '' END AS sg
      FROM b
    ) TO '{recp}' (FORMAT parquet)""")
    log("rec done")
krp = p("cache", "kr.parquet").replace("\\", "/")
if not os.path.exists(krp):
    con.sql(f"""COPY (SELECT appl_no, appl_year::INT appl_year, ksic, applicant, prio_cc, title
        FROM read_parquet('{META}') WHERE cc='KR') TO '{krp}' (FORMAT parquet)""")
    log("kr done")
famp = p("cache", "fam_link.parquet").replace("\\", "/")
if not os.path.exists(famp):
    con.sql(f"""COPY (
      SELECT DISTINCT cc, appl_no, substr(tok, 3) AS kr_appl_no FROM (
        SELECT cc, appl_no, unnest(string_split(family, '|')) AS tok
        FROM read_parquet('{META}') WHERE cc <> 'KR' AND family IS NOT NULL AND family LIKE '%KR%')
      WHERE regexp_full_match(tok, 'KR(10|20)[0-9]{{11}}')
    ) TO '{famp}' (FORMAT parquet)""")
    log("fam done")

# 연계표 그룹 배정(라벨공간 27개 그룹 제약 argmax) — IPC 키 조합 단위로 한 번만
cgp = p("cache", "ipc_cw.parquet")
if not os.path.exists(cgp):
    keys = con.sql(f"SELECT DISTINCT sg, mg, sc FROM '{recp}'").df()
    tabs = make_lookup(load_crosswalk(), constrained=True)
    keys["cw"] = assign_keys(keys.sg, keys.mg, keys.sc, tabs)
    keys.to_parquet(cgp, index=False)
    log("ipc keys", len(keys), "assigned", keys.cw.notna().mean())
print(con.sql(f"SELECT cc, count(*), min(pyear), max(pyear), avg((mg='')::INT) FROM '{recp}' GROUP BY cc").fetchall())
print(con.sql(f"SELECT cc, count(*), count(DISTINCT appl_no) FROM '{famp}' GROUP BY cc").fetchall())
log("end")

# -*- coding: utf-8 -*-
"""누수 점검용 '원점 시점(point-in-time) 해외패밀리' 집계.
KR 출원의 family 문자열(수집시점 스냅샷)에서 연도가 박힌 해외 공개번호(US·JP·WO + 4자리 연도)만 골라 '첫 해외 공개연도 fpy'를 뽑는다.
원점 t0 의 PIT 플래그 = 외국 우선권 출원(dom=0, 출원 시 확정) 또는 fpy ≤ t0. EP·CN 처럼 연도 없는 번호는 버린다(보수적 과소집계).
출력 cache/kr_fam_pit.parquet (ksic, y, dom, fpy, n)"""
import duckdb
from n1lib import CACHE, META
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=2")
con.execute(f"""COPY (
 WITH a AS (SELECT ksic, CAST(appl_year AS INT) y, appl_no,
             CASE WHEN prio_cc IS NULL OR split_part(prio_cc,'|',1)='KR' THEN 1 ELSE 0 END dom,
             regexp_extract_all(coalesce(family,''), '(?:US|JP|WO)((?:19|20)\\d\\d)\\d{{6,7}}', 1) yrs
            FROM read_parquet('{META}') WHERE cc='KR' AND appl_year BETWEEN 1995 AND 2021)
 SELECT ksic, y, dom, CASE WHEN len(yrs)>0 THEN CAST(list_min(yrs) AS INT) END fpy, COUNT(DISTINCT appl_no) n
 FROM a GROUP BY 1,2,3,4) TO '{CACHE}/kr_fam_pit.parquet' (FORMAT parquet)""")
print(con.sql(f"""select dom, sum(n) n, sum(case when fpy is not null then n end) with_year, avg(fpy - y) lag_mean,
  sum(case when fpy - y <= 2 then n end)/sum(case when fpy is not null then n end) within2 from '{CACHE}/kr_fam_pit.parquet' group by 1""").df())

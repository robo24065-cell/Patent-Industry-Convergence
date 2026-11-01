# -*- coding: utf-8 -*-
"""KR 출원인 동학: (ksic, 출원연도) 고유 출원인 수, 해당 KSIC 신규 출원인 수(그 KSIC 최초 출원연도=y). 출원연도 ≤2022."""
import duckdb, os
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = os.path.join(B, "_ideas_R2", "industry", "cache")
M = os.path.join(B, "_shared", "meta", "*.parquet").replace("\\", "/")
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=2")
con.execute(f"""
COPY (
WITH a AS (SELECT ksic, cc, CAST(appl_year AS INT) y, trim(split_part(applicant,'|',1)) ap
           FROM read_parquet('{M}') WHERE appl_year BETWEEN 1995 AND 2022 AND applicant IS NOT NULL AND cc IN ('KR','CN','US'))
, f AS (SELECT ksic, cc, ap, MIN(y) fy FROM a GROUP BY 1,2,3)
SELECT a.ksic, a.cc, a.y, COUNT(DISTINCT a.ap) n_ap, COUNT(DISTINCT CASE WHEN f.fy=a.y THEN a.ap END) n_new
FROM a JOIN f USING (ksic, cc, ap) GROUP BY 1,2,3) TO '{OUT}/app_kcy.parquet' (FORMAT parquet)""")
print(con.sql(f"select cc, count(*), sum(n_new) from '{OUT}/app_kcy.parquet' group by 1"))

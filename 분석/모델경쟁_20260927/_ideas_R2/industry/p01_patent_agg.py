# -*- coding: utf-8 -*-
"""제공 융합데이터(5개국 × KSIC) → (ksic, cc, 출원연도) 고유 출원 수 집계. 특허 피처는 원점 t0에서 출원연도 ≤ t0-1만 쓴다(공개지연)."""
import duckdb, os
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = os.path.join(B, "_ideas_R2", "industry", "cache"); os.makedirs(OUT, exist_ok=True)
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=2")
M = os.path.join(B, "_shared", "meta", "*.parquet").replace("\\", "/")
con.execute(f"""
COPY (
  SELECT ksic, cc, CAST(appl_year AS INT) AS y, COUNT(DISTINCT appl_no) AS n,
         COUNT(DISTINCT CASE WHEN prio_cc IS NOT NULL AND split_part(prio_cc,'|',1) <> cc THEN appl_no END) AS n_forprio
  FROM read_parquet('{M}') WHERE appl_year BETWEEN 1995 AND 2023
  GROUP BY 1,2,3) TO '{OUT}/pat_kcy.parquet' (FORMAT parquet)""")
# 서브클래스 단위(기술 다양성·신흥 서브클래스용)
con.execute(f"""
COPY (
  SELECT ksic, cc, CAST(appl_year AS INT) AS y, substr(replace(main_ipc,' ',''),1,4) AS sc, COUNT(DISTINCT appl_no) AS n
  FROM read_parquet('{M}') WHERE appl_year BETWEEN 1995 AND 2023 AND main_ipc IS NOT NULL
  GROUP BY 1,2,3,4) TO '{OUT}/pat_kcys.parquet' (FORMAT parquet)""")
print(con.sql(f"select cc, count(*), sum(n) from '{OUT}/pat_kcy.parquet' group by 1"))

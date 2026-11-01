# -*- coding: utf-8 -*-
"""한국발(우선권 KR) 해외출원: (ksic, 출원국 cc∈US·EP·CN·JP, 출원연도) 우선권 첫 국가별 고유 출원 수. 한국 기술의 '해외 진출 점유' 지표."""
import duckdb, os
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = os.path.join(B, "_ideas_R2", "industry", "cache"); M = os.path.join(B, "_shared", "meta", "*.parquet").replace("\\", "/")
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=2")
con.execute(f"""COPY (SELECT ksic, cc, CAST(appl_year AS INT) y, split_part(prio_cc,'|',1) pc, COUNT(DISTINCT appl_no) n
  FROM read_parquet('{M}') WHERE appl_year BETWEEN 1995 AND 2022 AND cc <> 'KR' AND prio_cc IS NOT NULL
  GROUP BY 1,2,3,4) TO '{OUT}/prio_kcy.parquet' (FORMAT parquet)""")
print(con.sql(f"select cc, pc, sum(n) s from '{OUT}/prio_kcy.parquet' where pc in ('KR','US','JP','CN','DE') group by 1,2 order by 1, s desc"))

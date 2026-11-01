# -*- coding: utf-8 -*-
"""N1 추가 특허 집계(제공 5개국 meta, duckdb, 읽기 전용).
1) pat_pub.parquet: (ksic, cc, 출원연도 y, 공개연도 py=coalesce(공개일, 등록일)) 고유 출원 수 → '공개일 ≤ t0' 조건 비교용
2) kr_qual.parquet: KR 출원 (ksic, y, dom=국내우선권/무우선권, ffam=해외 패밀리 보유, ry=등록연도) 고유 출원 수
   ※ family 는 수집시점 스냅샷이라 사후 확장 가능 → 시차 ≥3 로만 쓰고 한계로 기록
3) cn_type.parquet: CN 출원 (ksic, y, typ=1 발명/2 실용신안/3 디자인) 고유 출원 수
출원연도 1995~2021만 집계(개발·최종 피처 모두 ≤2020 필요)."""
import duckdb, os, time
from n1lib import CACHE, META
con = duckdb.connect(); con.execute("SET memory_limit='3GB'; SET threads=2")
t = time.time()
con.execute(f"""COPY (
  SELECT ksic, cc, CAST(appl_year AS INT) y,
         CAST(substr(coalesce(pub_date, rgst_date), 1, 4) AS INT) py, COUNT(DISTINCT appl_no) n
  FROM read_parquet('{META}') WHERE appl_year BETWEEN 1995 AND 2021 GROUP BY 1,2,3,4) TO '{CACHE}/pat_pub.parquet' (FORMAT parquet)""")
print("pat_pub", round(time.time()-t, 1)); t = time.time()
con.execute(f"""COPY (
  SELECT ksic, CAST(appl_year AS INT) y,
         CASE WHEN prio_cc IS NULL OR split_part(prio_cc,'|',1)='KR' THEN 1 ELSE 0 END dom,
         CASE WHEN regexp_matches(coalesce(family,''), '(^|\\|)(US|EP|JP|CN|WO)') THEN 1 ELSE 0 END ffam,
         CAST(substr(rgst_date,1,4) AS INT) ry, COUNT(DISTINCT appl_no) n
  FROM read_parquet('{META}') WHERE cc='KR' AND appl_year BETWEEN 1995 AND 2021 GROUP BY 1,2,3,4,5) TO '{CACHE}/kr_qual.parquet' (FORMAT parquet)""")
print("kr_qual", round(time.time()-t, 1)); t = time.time()
con.execute(f"""COPY (
  SELECT ksic, CAST(appl_year AS INT) y,
         CASE WHEN length(split_part(appl_no,'.',1))>=12 THEN substr(appl_no,5,1) ELSE substr(appl_no,3,1) END typ,
         COUNT(DISTINCT appl_no) n
  FROM read_parquet('{META}') WHERE cc='CN' AND appl_year BETWEEN 1995 AND 2021 GROUP BY 1,2,3) TO '{CACHE}/cn_type.parquet' (FORMAT parquet)""")
print("cn_type", round(time.time()-t, 1))
print(con.sql(f"select typ, sum(n) from '{CACHE}/cn_type.parquet' group by 1 order by 2 desc").df())
print(con.sql(f"select dom, ffam, sum(n) n, sum(case when ry is not null then n end) reg from '{CACHE}/kr_qual.parquet' group by 1,2").df())
print(con.sql(f"select cc, sum(case when py-y<=1 then n end)/sum(n) p1, sum(case when py-y<=2 then n end)/sum(n) p2, sum(case when py is null then n end)/sum(n) pnull from '{CACHE}/pat_pub.parquet' where y between 2005 and 2016 group by 1").df())

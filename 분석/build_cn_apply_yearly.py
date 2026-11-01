# -*- coding: utf-8 -*-
"""
중국(CN) 특허 출원 연도별 건수 원자료 집계 — 세세분류(KSIC 5자리) × 출원연도
목적: '중국 특허출원 급증'이 '대중 수입침투 상승'보다 시간적으로 선행하는지 검정할
      선행지표(lead) 시계열을 원자료(1,514만행)에서 직접 생성.

주의:
  - 융합데이터_*/*.csv 는 TSV(sep='\t'), encoding='utf-8-sig', 초록/청구항에 따옴표·개행 다수
  - 파일명 = 세세분류 코드이지만, 안전하게 '세세분류 코드' 컬럼값을 사용
  - 같은 출원번호가 한 산업 내 중복 기재될 수 있으므로 count(distinct 출원번호) 도 함께 산출
"""
import os
import duckdb

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, "분석", "fusion")
os.makedirs(OUT, exist_ok=True)
src = os.path.join(BASE, "공모전데이터", "융합데이터_*", "*.csv").replace(os.sep, "/")

d = duckdb.connect()
d.execute("SET threads TO 6; SET preserve_insertion_order=false;")
rd = ("read_csv('" + src + "', delim='\t', quote='\"', escape='\"', header=true, "
      "ignore_errors=true, sample_size=20000)")

# 전체 국가 × 산업 × 연도 (CN 뿐 아니라 5개국 모두 — 정규화(점유율)에 필요)
q = f"""
SELECT trim(CAST("세세분류 코드" AS VARCHAR)) AS ksic,
       trim("국가코드")                       AS ctry,
       CAST("출원연도" AS INT)                AS year,
       count(*)                                AS n_row,
       count(DISTINCT trim(CAST("출원번호" AS VARCHAR))) AS n_apply
FROM {rd}
WHERE "출원연도" IS NOT NULL
  AND TRY_CAST("출원연도" AS INT) BETWEEN 1990 AND 2025
GROUP BY 1, 2, 3
"""
print("스캔 시작 (약 31GB)...", flush=True)
df = d.execute(q).fetchdf()
print(f"완료: {len(df):,}행", flush=True)

df.to_csv(os.path.join(OUT, "apply_yearly_all.csv"), index=False, encoding="utf-8-sig")

cn = df[df.ctry == "CN"][["ksic", "year", "n_apply", "n_row"]].sort_values(["ksic", "year"])
cn.to_csv(os.path.join(OUT, "cn_apply_yearly.csv"), index=False, encoding="utf-8-sig")
print(f"CN: {len(cn):,}행, 산업 {cn.ksic.nunique()}개, "
      f"{cn.year.min()}~{cn.year.max()}, 총 {cn.n_apply.sum():,}건", flush=True)
print(f"중복(행수-고유출원수) 비율: {1 - cn.n_apply.sum()/cn.n_row.sum():.4%}", flush=True)

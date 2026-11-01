# -*- coding: utf-8 -*-
"""
융합데이터 고유자산 추출 — 지금까지 미사용이던 것들
사전 등록 (2026-08-12, 결과 보기 전):
  [A] 가치 층화: 패밀리 보유 여부 → PADI_hi(고가치) / PADI_lo
      가설: 해외출원 비용까지 지불한 특허의 포기 = 훨씬 강한 신호 → 신호대잡음비 개선
  [B] 경쟁압력 축: 산업×연도 5개국 출원 패널 → CN 출원 증가율 = 중국발 경쟁압력
      가설: 포기의 원인 중 '경쟁'을 분리 → 교란 제거로 잔여 신호 선명화
  [C] 기술 다각도: 산업×IPC4 단위 포기율 → 확산지수(breadth) 재료
"""
import duckdb, os
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
WORK = os.path.join(BASE, "공모전데이터", "작업데이터")
OUT = os.path.join(BASE, "분석", "fusion")
os.makedirs(OUT, exist_ok=True)
src = os.path.join(BASE, "공모전데이터", "융합데이터_*", "*.csv").replace(os.sep, "/")
log = lambda m: print(m, flush=True)

d = duckdb.connect()
d.execute("SET threads TO 6; SET preserve_insertion_order=false;")
rd = ("read_csv('" + src + "', delim='\t', quote='\"', escape='\"', header=true, "
      "ignore_errors=true, sample_size=20000)")

# ── [A] KR 등록특허 가치속성 (패밀리·우선권·IPC) ────────────────
log("[A] KR 등록특허 가치속성 추출 중...")
d.execute(f"""COPY (
  SELECT trim(CAST("등록번호" AS VARCHAR)) AS rgst_no,
         trim(CAST("세세분류 코드" AS VARCHAR)) AS ksic,
         CASE WHEN "패밀리 출원번호" IS NOT NULL AND trim("패밀리 출원번호")<>'' THEN 1 ELSE 0 END AS has_family,
         CASE WHEN "우선권 주장 출원번호" IS NOT NULL AND trim("우선권 주장 출원번호")<>'' THEN 1 ELSE 0 END AS has_prio,
         substr(trim("메인 ipc"), 1, 4) AS ipc4,
         length(coalesce("IPC",'')) - length(replace(coalesce("IPC",''), '|', '')) + 1 AS n_ipc
  FROM {rd}
  WHERE trim("국가코드")='KR' AND "등록번호" IS NOT NULL AND trim(CAST("등록번호" AS VARCHAR))<>''
) TO '{os.path.join(OUT, "kr_value_attrs.parquet").replace(os.sep, "/")}' (FORMAT PARQUET);""")
a = d.execute(f"SELECT count(*) n, avg(has_family) f, avg(has_prio) p FROM read_parquet('{os.path.join(OUT, 'kr_value_attrs.parquet').replace(os.sep, '/')}')").fetchdf()
log(f"    {int(a.n[0]):,}건 | 패밀리 {a.f[0]*100:.1f}% | 우선권 {a.p[0]*100:.1f}%")

# ── [B] 산업×연도×국가 출원 패널 (경쟁압력) ─────────────────────
log("[B] 산업×연도×국가 출원 패널 구축 중 (1,514만건 스캔)...")
comp = d.execute(f"""
  SELECT trim(CAST("세세분류 코드" AS VARCHAR)) AS ksic,
         trim("국가코드") AS ctry,
         CAST("출원연도" AS INT) AS yr,
         count(*) AS n
  FROM {rd}
  WHERE "출원연도" IS NOT NULL AND CAST("출원연도" AS INT) BETWEEN 2000 AND 2023
  GROUP BY 1,2,3""").fetchdf()
comp.to_csv(os.path.join(OUT, "industry_country_year.csv"), index=False, encoding="utf-8-sig")
log(f"    {len(comp):,}행 (산업 {comp.ksic.nunique()} × 국가 {comp.ctry.nunique()} × 연도 {comp.yr.nunique()})")

# 경쟁압력 지표 산출
piv = comp.pivot_table(index=["ksic", "yr"], columns="ctry", values="n", fill_value=0).reset_index()
piv.columns.name = None
for c in ("CN", "US", "JP", "EP", "KR"):
    if c not in piv:
        piv[c] = 0
piv["tot"] = piv[["CN", "US", "JP", "EP", "KR"]].sum(axis=1)
piv["cn_share"] = piv.CN / piv.tot.replace(0, pd.NA)
piv["kr_share"] = piv.KR / piv.tot.replace(0, pd.NA)
piv = piv.sort_values(["ksic", "yr"])
piv["cn_growth"] = piv.groupby("ksic").CN.pct_change() * 100
piv["cn_share_chg"] = piv.groupby("ksic").cn_share.diff() * 100
piv["kr_share_chg"] = piv.groupby("ksic").kr_share.diff() * 100
# 한국의 상대적 지위: KR/(US+JP+EP) — 선진국 대비
piv["kr_rel_adv"] = piv.KR / (piv.US + piv.JP + piv.EP).replace(0, pd.NA)
piv.to_csv(os.path.join(OUT, "competition_panel.csv"), index=False, encoding="utf-8-sig")
log(f"    경쟁압력 패널 저장 (CN비중·증가율·한국상대지위)")

log("\n[경쟁압력 요약 — 2010~2023 평균]")
s = piv[(piv.yr >= 2010) & (piv.yr <= 2023)].groupby("ksic").agg(
    cn_share=("cn_share", "mean"), kr_share=("kr_share", "mean"),
    cn_growth=("cn_growth", "mean"), kr_rel=("kr_rel_adv", "mean")).reset_index()
log("  중국 비중 최상위 3:\n" + s.nlargest(3, "cn_share").round(3).to_string(index=False))
log("  중국 비중 최하위 3:\n" + s.nsmallest(3, "cn_share").round(3).to_string(index=False))
s.to_csv(os.path.join(OUT, "competition_agg.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {OUT}")

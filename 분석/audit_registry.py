# -*- coding: utf-8 -*-
"""
등록원부 수집 데이터 품질 감사
────────────────────────────────────────────────────────────────
지금까지 건수만 셌지 내용을 점검한 적이 없다. 주제와 무관하게 필요한 작업이다.
특히 확인할 것:
  · getPayHistory 폴백(1차 오퍼레이션 한도 소진 시 사용)이 품질을 떨어뜨렸는가
  · 논리적으로 불가능한 값이 있는가 (최종연차 > 등록경과연수 등)
  · 실용신안/특허의 존속기간 상한(10년/20년)이 데이터에 반영되어 있는가
  · 원본 JSON 대비 파싱이 정확한가
"""
import sqlite3, gzip, json, io, sys
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = BASE + r"\수집파이프라인\registry.sqlite"
log = lambda m: print(m, flush=True)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
d = pd.read_sql("SELECT rgst_no,ksic,appl_no,reg_date,exp_date,term_date,"
                "term_cause,claim_count,last_annl,source FROM registry WHERE status='OK'", con)
log(f"[감사 대상] {len(d):,}건\n")

# ── 1. 소스별 품질
log("[1] 소스별 — getPayHistory 폴백이 품질을 떨어뜨렸는가")
for s, g in d.groupby("source"):
    ln = pd.to_numeric(g.last_annl, errors="coerce")
    rd = g.reg_date.notna() & (g.reg_date.astype(str).str.strip() != "")
    cc = pd.to_numeric(g.claim_count, errors="coerce")
    log(f"  {s:12s} {len(g):7,}건 | 연차유효 {ln.notna().mean()*100:5.1f}% "
        f"| 등록일유효 {rd.mean()*100:5.1f}% | 청구항유효 {cc.notna().mean()*100:5.1f}% "
        f"| 연차중앙 {ln.median():.0f}")

# ── 2. 논리 정합성
log("\n[2] 논리 정합성 검사")
d["ln"] = pd.to_numeric(d.last_annl, errors="coerce")
d["ry"] = pd.to_numeric(d.reg_date.astype(str).str[:4], errors="coerce")
d["ay"] = pd.to_numeric(d.appl_no.astype(str).str[:4], errors="coerce")
age = 2026 - d.ry
chk = [
    ("최종연차 > 등록경과연수+1 (불가능)", (d.ln > age + 1) & d.ln.notna() & d.ry.notna()),
    ("최종연차 < 1 (불가능)", (d.ln < 1) & d.ln.notna()),
    ("등록연도 < 출원연도 (불가능)", (d.ry < d.ay) & d.ry.notna() & d.ay.notna()),
    ("등록연도 이상치 (<1990 or >2026)", ((d.ry < 1990) | (d.ry > 2026)) & d.ry.notna()),
]
for lab, m in chk:
    log(f"  {lab:36s} {int(m.sum()):7,} ({m.mean()*100:5.2f}%)")
log(f"  {'등록번호 중복':36s} {int(d.rgst_no.duplicated().sum()):7,}")
log(f"  {'출원번호 중복':36s} {int(d.appl_no.duplicated().sum()):7,}")

# ── 3. 권리 유형별 존속기간
log("\n[3] 권리 유형별 — 존속기간 상한이 데이터에 반영되는가")
d["typ"] = np.where(d.rgst_no.astype(str).str.startswith("2"), "실용신안", "특허")
for t, g in d.groupby("typ"):
    ln = pd.to_numeric(g.last_annl, errors="coerce").dropna()
    log(f"  {t:6s} {len(g):7,}건 | 연차 중앙 {ln.median():.0f} 평균 {ln.mean():4.1f} "
        f"| p95 {ln.quantile(.95):.0f} 최대 {ln.max():.0f}")
log("  ※ 실용신안 존속 10년, 특허 20년 — 상한이 다르게 나와야 정상")

# ── 4. 산업별 커버리지와 연차 분포
log("\n[4] 산업별 연차 중앙값 — 산업 간 실질 차이가 있는가")
g = d[d.ln.notna()].groupby("ksic").agg(n=("rgst_no", "size"), med=("ln", "median"),
                                        mean=("ln", "mean")).reset_index()
g = g[g.n >= 3000].sort_values("med")
log(f"  {len(g)}개 산업 | 연차 중앙값 범위 {g.med.min():.0f} ~ {g.med.max():.0f}")
log("  [짧은 5]  " + " ".join(f"{r.ksic}({r.med:.0f})" for r in g.head(5).itertuples()))
log("  [긴 5]    " + " ".join(f"{r.ksic}({r.med:.0f})" for r in g.tail(5).itertuples()))

# ── 5. 원본 대조
log("\n[5] 원본 JSON 대조 (무작위 5건) — 파싱 정확성")
rows = con.execute("SELECT rgst_no,last_annl,claim_count,reg_date,raw_gz FROM registry "
                   "WHERE status='OK' AND raw_gz IS NOT NULL ORDER BY RANDOM() LIMIT 5").fetchall()
for rn, la, cc, rd, gz in rows:
    try:
        j = json.loads(gzip.decompress(gz).decode("utf-8"))
        b = j.get("response", {}).get("body", {}) if isinstance(j, dict) else {}
        it = b.get("items") or b.get("item") or {}
        if isinstance(it, dict):
            it = it.get("item", it)
        if isinstance(it, list) and it:
            it = it[0]
        ks = list(it)[:10] if isinstance(it, dict) else []
        log(f"  {rn}: DB(연차={la}, 청구항={cc}, 등록일={rd})")
        log(f"    원본키: {ks}")
        if isinstance(it, dict):
            hit = {k: v for k, v in it.items()
                   if any(t in str(k).lower() for t in ("annl", "pay", "claim", "regi", "date"))}
            log(f"    관련값: {dict(list(hit.items())[:6])}")
    except Exception as e:
        log(f"  {rn}: 원본 파싱 실패 {type(e).__name__} {e}")

con.close()
log("\n[감사 종료]")

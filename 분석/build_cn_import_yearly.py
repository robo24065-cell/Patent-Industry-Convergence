# -*- coding: utf-8 -*-
"""
대(對)중국 수입침투 연도별 시계열 구축 — 2개 해상도

[A] KSIC 중분류(2자리) 해상도 · 2010~2025 · 개념적으로 정확한 '침투율'
    KOSIS 「제조업 국내공급동향」
      DT_1MS2006 수입점유비(업종별)            = 국내공급 중 수입 비중(%)
      DT_1MS2004 수입의 국가(권역)비중(업종별) = 수입 중 중국 비중(%), 분기 → 연평균
    cn_imp_pen = 수입점유비 × 중국비중 / 100  ( = 국내공급 중 중국산 비중, %p )

[B] KSIC 세세분류(5자리) 해상도 · 2000~2024 · 금액기반(별도 스크립트가 수집한 HS6 결합)
    fetch_hs6_cn_imports.py 산출물(hs6_import_yearly.csv) + ksic_hs6_map.csv 를 결합

주의: [A]는 우리 유효 클러스터(mid 13개)와 단위가 같다 → 횡단면 회귀의 통제변수로는
      within 변동이 0이라 부적합. 다만 '선행성(lead-lag)' 검정은 산업×연도 패널의
      시간 방향 변동으로 식별하므로 2자리 해상도라도 사용 가능하다. [B]가 5자리 보완재.
"""
import os, json, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(BASE, "분석", "fusion")
AX = os.path.join(BASE, "공모전데이터", "외부데이터", "제3축")
KEY = "ZWE4NzllN2FkYWIwMzVmYzBiNjk0N2M0ZWEzMzIxNjQ="
log = lambda m: print(m, flush=True)


def kosis(**p):
    u = ("https://kosis.kr/openapi/Param/statisticsParameterData.do?"
         + urllib.parse.urlencode(p, quote_via=urllib.parse.quote))
    for a in range(4):
        try:
            with urllib.request.urlopen(u, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            log(f"  KOSIS 재시도({a}): {e}")
            time.sleep(5)
    return None


# ── [A-1] 수입 중 중국비중 (분기 → 연평균) ─────────────────────
j = kosis(method="getList", apiKey=KEY, itmId="T1+", objL1="ALL", objL2="ALL",
          format="json", jsonVD="Y", prdSe="Q", startPrdDe="200001",
          endPrdDe="202604", orgId="101", tblId="DT_1MS2004")
q = pd.DataFrame(j)
q = q[q.C2 == "01"]                                  # C2=01 → 중국
q["mid"] = q.C1.str.extract(r"^C(\d{2})$")[0]
q = q.dropna(subset=["mid"])
q["year"] = q.PRD_DE.str[:4].astype(int)
q["val"] = pd.to_numeric(q.DT, errors="coerce")
cn_share = (q.dropna(subset=["val"]).groupby(["mid", "year"], as_index=False)
              .agg(cn_share_imp=("val", "mean"), nq=("val", "size")))
cn_share = cn_share[cn_share.nq == 4]                # 완전한 4분기만
log(f"[A-1] 수입 중 중국비중: 업종 {cn_share.mid.nunique()} × "
    f"{cn_share.year.min()}~{cn_share.year.max()} ({len(cn_share)}행)")

# ── [A-2] 수입점유비 (연간, 로컬 CSV 우선 + API 보강) ───────────
imp = pd.read_csv(os.path.join(AX, "수입점유비_업종별.csv"), encoding="utf-8-sig")
imp["mid"] = imp.C1.astype(str).str.extract(r"^C(\d{2})$")[0]
imp["year"] = imp.PRD_DE.astype(int)
imp["imp_share"] = pd.to_numeric(imp.DT, errors="coerce")
imp = imp.dropna(subset=["mid", "imp_share"])[["mid", "year", "imp_share"]]
log(f"[A-2] 수입점유비: 업종 {imp.mid.nunique()} × {imp.year.min()}~{imp.year.max()}")

A = imp.merge(cn_share[["mid", "year", "cn_share_imp"]], on=["mid", "year"], how="inner")
A["cn_imp_pen"] = A.imp_share * A.cn_share_imp / 100.0
A = A.sort_values(["mid", "year"]).reset_index(drop=True)
A["ksic_level"] = 2
A.to_csv(os.path.join(OUT, "cn_import_yearly.csv"), index=False, encoding="utf-8-sig")
log(f"[A] 저장 cn_import_yearly.csv : {len(A)}행, 업종 {A.mid.nunique()}, "
    f"{A.year.min()}~{A.year.max()}")
log("    대중침투율 상위(2024): " +
    str(A[A.year == 2024].nlargest(4, "cn_imp_pen")[["mid", "cn_imp_pen"]]
        .round(1).to_dict("records")))

# ── [B] HS6 → KSIC5 (별도 수집 완료 시) ─────────────────────────
hs_f = os.path.join(OUT, "hs6_import_yearly.csv")
map_f = os.path.join(OUT, "ksic_hs6_map.csv")
if os.path.exists(hs_f) and os.path.exists(map_f):
    hs = pd.read_csv(hs_f, encoding="utf-8-sig", dtype={"hs6": str})
    mp = pd.read_csv(map_f, encoding="utf-8-sig", dtype={"hs6": str, "k5": str})
    # HS6가 여러 KSIC에 걸치면 균등 분할(1/n) — 83%는 1:1이라 영향 작음
    w = mp.groupby("hs6").k5.transform("size")
    mp = mp.assign(wgt=1.0 / w)
    piv = hs.pivot_table(index=["hs6", "year"], columns="partner",
                         values="value", aggfunc="sum").reset_index()
    for c in ("CN", "WLD"):
        if c not in piv:
            piv[c] = np.nan
    m = mp.merge(piv, on="hs6", how="inner")
    m["CN"] = m.CN.fillna(0) * m.wgt
    m["WLD"] = m.WLD.fillna(0) * m.wgt
    B = (m.groupby(["k5", "year"], as_index=False)
           .agg(cn_imp_usd=("CN", "sum"), wld_imp_usd=("WLD", "sum"),
                n_hs6=("hs6", "nunique")))
    B["cn_share_imp"] = np.where(B.wld_imp_usd > 0,
                                 B.cn_imp_usd / B.wld_imp_usd * 100, np.nan)
    B = B.rename(columns={"k5": "ksic"}).sort_values(["ksic", "year"])
    B["ksic_level"] = 5
    B.to_csv(os.path.join(OUT, "cn_import_yearly_ksic5.csv"),
             index=False, encoding="utf-8-sig")
    log(f"[B] 저장 cn_import_yearly_ksic5.csv : {len(B)}행, "
        f"산업 {B.ksic.nunique()}, {B.year.min()}~{B.year.max()}")
else:
    log("[B] HS6 수집 결과 아직 없음 - fetch_hs6_cn_imports.py 완료 후 재실행")

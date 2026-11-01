# -*- coding: utf-8 -*-
"""
ECOS 기업경영분석 → 재무 통제변수 ctrl_financial.csv 생성

[설계 원칙 준수]
 1) 시간분할: 종속변수 dd_late 는 2017년 이후. 따라서 재무지표는 **2009~2016(전반기)** 평균만 사용.
    (ECOS 제11차 KSIC 표는 2009년부터 제공 → 전반기 창 2005~2016 중 2009~2016 이 최대 가용구간)
 2) 해상도: ECOS 업종 차원이 제조업 일부에서 KSIC 3자리(소분류)까지 제공됨.
    → k3(3자리) 기준으로 붙이고, 3자리가 없는 업종만 2자리로 폴백. 실제 해상도를 컬럼으로 기록.
 3) 두 표본(n=39 원자료 / n=23 k3집계) 모두 검정.

출력: 분석/fusion/ctrl_financial.csv (utf-8-sig)
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
RAW = os.path.join(BASE, "공모전데이터", "외부데이터", "재무")
FUS = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

PRE_LO, PRE_HI = 2009, 2016          # 전반기(2017년 이전) 창
SIZE = "A"                            # 기업규모: 종합

# ── 지표 정의: 컬럼명 → (통계표, ITEM_CODE3) ────────────────────────────────
IND = {
    "fin_debt_ratio":   ("501Y007", "707"),   # 부채비율 (%)
    "fin_equity_ratio": ("501Y007", "701"),   # 자기자본비율 (%)
    "fin_current_ratio":("501Y007", "702"),   # 유동비율 (%)
    "fin_borrow_dep":   ("501Y007", "710"),   # 차입금의존도 (%)
    "fin_opm":          ("501Y006", "611"),   # 매출액영업이익률 (%)
    "fin_icr":          ("501Y006", "627"),   # 이자보상비율 (%)
    "fin_int_rate":     ("501Y006", "623"),   # 차입금평균이자율 (%)
    "fin_rnd_sales":    ("501Y006", "615"),   # 연구개발비대매출액 (%)
    "fin_sales_growth": ("501Y005", "506"),   # 매출액증가율 (%)
}
FILES = {
    "501Y005": "ecos_501Y005_성장성지표.csv",
    "501Y006": "ecos_501Y006_손익지표.csv",
    "501Y007": "ecos_501Y007_자산자본지표.csv",
}

# ── KSIC 3자리 → ECOS 업종코드 매핑 (ECOS ITEM_NAME1 의 포괄범위를 직접 확인해 작성) ──
#    res: 3 = KSIC 3자리 단독, 3m = 3자리 병합군(여러 소분류 묶음), 2 = 2자리(중분류)만 존재
K3_MAP = {
    "152": ("C15",  2,  "C15 가죽, 가방 및 신발 (3자리 미제공)"),
    "202": ("C202", 3,  "C202 합성고무 및 플라스틱 물질"),
    "204": ("C204", 3,  "C204 기타 화학제품"),
    "212": ("C21",  2,  "C21 의료용 물질 및 의약품 (3자리 미제공)"),
    "239": ("C239", 3,  "C239 기타 비금속 광물제품"),
    "241": ("C241", 3,  "C241 1차 철강"),
    "242": ("C242", 3,  "C242 1차 비철금속"),
    "259": ("C25",  2,  "C25 금속가공제품 (3자리 미제공)"),
    "261": ("C262", 31, "C261,2 반도체 및 전자부품 (261+262 병합)"),
    "262": ("C262", 31, "C261,2 반도체 및 전자부품 (261+262 병합)"),
    "263": ("C263", 3,  "C263 컴퓨터 및 주변장치"),
    "264": ("C264", 3,  "C264 통신 및 방송 장비"),
    "265": ("C266", 31, "C265,6 영상·음향기기, 마그네틱 및 광학매체 (265+266 병합)"),
    "272": ("C27",  2,  "C27 의료, 정밀, 광학기기 및 시계 (3자리 미제공)"),
    "281": ("C281", 3,  "C281 전동기, 발전기 및 전기변환·공급·제어장치"),
    "285": ("C285", 3,  "C285 가정용 기기"),
    "289": ("C289", 31, "C282,4,9 기타 전기장비 (282+284+289 병합)"),
    "291": ("C291", 3,  "C291 일반 목적용 기계"),
    "292": ("C292", 3,  "C292 특수 목적용 기계"),
    "303": ("C303", 31, "C302-4 자동차 차체·트레일러 및 자동차 부품 (302~304 병합)"),
    "312": ("C319", 31, "C312,3,9 철도·항공기 및 그 외 기타 운송장비 (312+313+319 병합)"),
    "313": ("C319", 31, "C312,3,9 철도·항공기 및 그 외 기타 운송장비 (312+313+319 병합)"),
    "333": ("C33",  2,  "C33 기타 제품 제조업 (3자리 미제공)"),
}
# 중분류(2자리) 대조군 — 해상도 함정 실증용
MID_MAP = {m: f"C{m}" for m in
           ["15", "20", "21", "23", "24", "25", "26", "27", "28", "29", "30", "31", "33"]}


# ── 1. 원자료 적재 ─────────────────────────────────────────────────────────
def load_raw():
    frames = []
    for code, fn in FILES.items():
        d = pd.read_csv(os.path.join(RAW, fn), encoding="utf-8-sig",
                        dtype={"ITEM_CODE1": str, "ITEM_CODE2": str,
                               "ITEM_CODE3": str, "TIME": str})
        d["val"] = pd.to_numeric(d["DATA_VALUE"], errors="coerce")
        d["year"] = pd.to_numeric(d["TIME"], errors="coerce")
        frames.append(d[["STAT_CODE", "ITEM_CODE1", "ITEM_NAME1", "ITEM_CODE2",
                         "ITEM_CODE3", "ITEM_NAME3", "year", "val"]])
    return pd.concat(frames, ignore_index=True)


R = load_raw()
log(f"[1] ECOS 원자료 적재 {len(R):,}행 | 연도 {int(R.year.min())}~{int(R.year.max())}")
R = R[R.ITEM_CODE2 == SIZE]                      # 기업규모 종합만
log(f"    기업규모=종합(A) 필터 후 {len(R):,}행")

# 매핑에 쓰는 ECOS 코드가 실제 존재하는지 검증
avail = set(R.ITEM_CODE1.unique())
need = set(v[0] for v in K3_MAP.values()) | set(MID_MAP.values())
missing = need - avail
log(f"    필요 업종코드 {len(need)}개 중 결측 {len(missing)}개 {sorted(missing) if missing else ''}")
nm = R.drop_duplicates("ITEM_CODE1").set_index("ITEM_CODE1")["ITEM_NAME1"].to_dict()
log("    [매핑 감사] k3 → ECOS 업종 실제 명칭")
for k3, (ec, res, memo) in sorted(K3_MAP.items()):
    log(f"      {k3} → {ec:5s} res={str(res):2s} | {nm.get(ec, '없음')}")


# ── 2. 전반기(2009~2016) 평균 산출 ────────────────────────────────────────
def pre_mean(ecos_code, stat, item3):
    s = R[(R.ITEM_CODE1 == ecos_code) & (R.STAT_CODE == stat) &
          (R.ITEM_CODE3 == item3) & (R.year >= PRE_LO) & (R.year <= PRE_HI)]
    v = s["val"].dropna()
    return (v.mean(), len(v)) if len(v) else (np.nan, 0)


def icr_below100(ecos_code):
    """이자보상비율 < 100% (영업이익 < 이자비용) 인 해의 비중 — 한계기업 위험 대리변수"""
    s = R[(R.ITEM_CODE1 == ecos_code) & (R.STAT_CODE == "501Y006") &
          (R.ITEM_CODE3 == "627") & (R.year >= PRE_LO) & (R.year <= PRE_HI)]
    v = s["val"].dropna()
    return (v < 100).mean() if len(v) else np.nan


rows = []
for k3, (ec, res, memo) in K3_MAP.items():
    r = {"k3": k3, "fin_src_code": ec, "fin_res": res, "fin_src_name": memo}
    for col, (stat, it) in IND.items():
        m, n = pre_mean(ec, stat, it)
        r[col] = m
        r[f"_n_{col}"] = n
    r["fin_icr_low"] = icr_below100(ec)
    rows.append(r)
F = pd.DataFrame(rows)

# 중분류(2자리) 버전 — 대조/폴백용
mrows = []
for mid, ec in MID_MAP.items():
    r = {"mid": mid, "fin_mid_src": ec}
    for col, (stat, it) in IND.items():
        r[f"{col}_mid"], _ = pre_mean(ec, stat, it)
    mrows.append(r)
M = pd.DataFrame(mrows)

log(f"\n[2] 전반기 {PRE_LO}~{PRE_HI} 평균 산출 | k3표 {F.shape} / mid표 {M.shape}")
ncols = [c for c in F.columns if c.startswith("_n_")]
log("    지표별 사용 연도수(최대 8) 최소/최대: " +
    ", ".join(f"{c[3:]}={int(F[c].min())}~{int(F[c].max())}" for c in ncols))


# ── 3. predictive_agg 와 병합 검증 ────────────────────────────────────────
P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
log(f"\n[3] predictive_agg.csv {P.shape} | k3 {P.k3.nunique()}개 | mid {P.mid.nunique()}개")

Z = P.merge(F.drop(columns=ncols), on="k3", how="left").merge(M, on="mid", how="left")
log(f"    merge 후 {Z.shape[0]}행 (원 39행 유지: {Z.shape[0] == 39})")
for c in list(IND) + ["fin_icr_low"]:
    log(f"    {c:20s} 비결측 {Z[c].notna().sum():2d}/39 (k3 {Z.dropna(subset=[c]).k3.nunique()}/23)")

# 해상도 진단
log(f"\n    [해상도 진단] fin_src_code 고유값 {Z.fin_src_code.nunique()}개 "
    f"(mid 13개 / k3 23개 대비)")
res_cnt = Z.drop_duplicates("k3").fin_res.value_counts().sort_index()
log(f"    k3 23개 중 3자리단독={int(res_cnt.get(3,0))} 3자리병합군={int(res_cnt.get(31,0))} "
    f"2자리폴백={int(res_cnt.get(2,0))}")
wi = Z.groupby("mid").fin_src_code.nunique()
log(f"    mid 내부 within 변동 있는 중분류: {int((wi > 1).sum())}/13 "
    f"→ {dict(wi[wi > 1])}")


# ── 4. 기준모형 + 재무통제 회귀 (n=39 / n=23 양쪽) ────────────────────────
def report(d, tag, groups):
    base = smf.ols("dd_late ~ padi_e + cn_growth_e", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    log(f"\n  [{tag}] n={int(base.nobs)} 클러스터={groups.nunique()}")
    log(f"    기준식        β_padi={base.params.padi_e:+8.2f} "
        f"z={base.tvalues.padi_e:+6.2f} p={base.pvalues.padi_e:.4f} R²={base.rsquared:.3f}")
    out = []
    for c in list(IND) + ["fin_icr_low"]:
        dd = d.dropna(subset=[c])
        if dd[c].std() == 0 or dd[c].notna().sum() < 10:
            log(f"    +{c:18s} 변동없음/표본부족 — 생략")
            continue
        g = groups.loc[dd.index]
        b0 = smf.ols("dd_late ~ padi_e + cn_growth_e", data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        f2 = smf.ols(f"dd_late ~ padi_e + cn_growth_e + {c}", data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        chg = (f2.params.padi_e - b0.params.padi_e) / abs(b0.params.padi_e) * 100
        log(f"    +{c:18s} β_padi={f2.params.padi_e:+8.2f} "
            f"z={f2.tvalues.padi_e:+6.2f} p={f2.pvalues.padi_e:.4f} "
            f"| β_ctrl={f2.params[c]:+8.3f} p={f2.pvalues[c]:.3f} "
            f"| R²={f2.rsquared:.3f} | Δβ_padi={chg:+5.1f}%")
        out.append(c)
    # 전부 동시 투입
    keep = [c for c in ["fin_debt_ratio", "fin_opm", "fin_icr_low", "fin_borrow_dep"] if c in out]
    if keep:
        dd = d.dropna(subset=keep)
        g = groups.loc[dd.index]
        f3 = smf.ols("dd_late ~ padi_e + cn_growth_e + " + " + ".join(keep), data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": g})
        log(f"    +[4종 동시]        β_padi={f3.params.padi_e:+8.2f} "
            f"z={f3.tvalues.padi_e:+6.2f} p={f3.pvalues.padi_e:.4f} R²={f3.rsquared:.3f}")
    return base


log("\n[4] 기준모형에 재무통제 1개씩 추가 (cluster by mid)")
D39 = Z.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).reset_index(drop=True)
report(D39, "n=39 원자료", D39.mid)

agg = {"dd_late": ("dd_late", "first"), "padi_e": ("padi_e", "mean"),
       "cn_growth_e": ("cn_growth_e", "mean"), "mid": ("mid", "first")}
for c in list(IND) + ["fin_icr_low"]:
    agg[c] = (c, "first")          # k3 단위 상수이므로 first
A23 = D39.groupby("k3").agg(**agg).reset_index()
report(A23, "n=23 k3집계", A23.mid)


# ── 5. PADI 와 재무지표의 상관 (허위상관 반론 직접 점검) ─────────────────
log("\n[5] PADI(전반기) vs 재무지표 상관 — '돈 없는 산업' 반론 직접 점검")
for c in list(IND) + ["fin_icr_low"]:
    dd = D39.dropna(subset=[c, "padi_e"])
    if len(dd) < 10:
        continue
    log(f"    corr(padi_e, {c:18s}) n39 = {dd.padi_e.corr(dd[c]):+.3f} "
        f"(ρ={dd.padi_e.corr(dd[c], method='spearman'):+.3f}) | "
        f"n23 = {A23.padi_e.corr(A23[c]):+.3f}")


# ── 6. 저장 ───────────────────────────────────────────────────────────────
cols = ["ksic", "k3", "mid", "fin_src_code", "fin_src_name", "fin_res"] + \
       list(IND) + ["fin_icr_low"] + [f"{c}_mid" for c in IND] + ["fin_mid_src"]
OUTF = os.path.join(FUS, "ctrl_financial.csv")
Z[cols].to_csv(OUTF, index=False, encoding="utf-8-sig")
log(f"\n[6] 저장 → {OUTF}  {Z[cols].shape}")
log(Z[["ksic", "k3", "fin_src_code", "fin_debt_ratio", "fin_opm", "fin_icr",
       "fin_borrow_dep", "fin_icr_low"]].to_string(index=False))

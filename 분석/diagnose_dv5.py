# -*- coding: utf-8 -*-
"""
왜 5자리 종속변수에서 결과가 사라지는가 — 원인 분해
────────────────────────────────────────────────────────────────
관측:
  기존 DV(월별 실질 생산지수, 3자리)  : β=-38.70, R²=0.333, wild-p=0.005
  신규 DV(연별 명목 출하액, 5자리)    : β= -2.40, R²=0.004, wild-p=0.854
  그런데 k3 고정효과에서는           : β=-28.37, p=0.034 (유의)

경합 가설 3개를 분리한다.
  [H1] 측정 차이 — 실질 생산지수 vs 명목 출하액은 다른 현상을 잰다
  [H2] 빈도 차이 — 월별 96점 vs 연별 8점. 연별은 낙폭을 과소·거칠게 잰다
  [H3] 해상도 차이 — PADI 는 소분류 수준 현상과 관계있고 세세분류 수준과는 무관하다

가르는 방법:
  · 신규 DV 를 k3 로 집계해 기존 DV 와 같은 해상도로 맞춘 뒤 재검정
      → 여기서 신호가 살아나면 H3 (해상도), 안 살아나면 H1/H2 (측정·빈도)
  · 기존 DV 를 연별로 다운샘플해 같은 빈도로 맞춘 뒤 재검정
      → 여기서 신호가 죽으면 H2 (빈도)
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
log  = lambda m: print(m, flush=True)

D = pd.read_csv(os.path.join(FUS, "predictive_agg_dv5.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
W = pd.read_csv(os.path.join(FUS, "mm_panel5.csv"), encoding="utf-8-sig", dtype={"ksic": str})
DVS = ["dd5_late", "dd5s_late", "dd5v_late", "dd5vs_late"]


def maxdd(s):
    s = np.asarray(s, float); s = s[~np.isnan(s)]
    if len(s) < 3 or np.nanmax(s) <= 0: return np.nan
    return float(np.min(s / np.maximum.accumulate(s) - 1.0) * 100)


def fit(d, y, x="padi_e", extra="+ cn_growth_e", cl="mid"):
    f = smf.ols(f"{y} ~ {x} {extra}", data=d).fit(cov_type="cluster", cov_kwds={"groups": d[cl]})
    return f.params[x], f.pvalues[x], f.rsquared, len(d)


# ══════════════════════════════════════════════════════════════
log("[H3 검정] 신규 DV 를 k3 로 집계 — 기존 DV 와 같은 해상도로 맞춘다")
log("  집계 방식: 소속 세세분류의 출하액을 합산한 뒤 낙폭 계산 (진짜 소분류 시계열)")
W["k3"] = W.ksic.str[:3]
tgt3 = set(D.k3)
A = W[W.k3.isin(tgt3)].groupby(["k3", "year"], as_index=False).agg(
    shipment=("shipment", "sum"), value_added=("value_added", "sum"))
tot = A.groupby("year").shipment.transform("sum")
A["ship_share"] = A.shipment / tot * 100
rows = []
for k, g in A.groupby("k3"):
    g = g.sort_values("year")
    e, l = g[g.year < 2017], g[g.year >= 2017]
    if len(e) >= 4 and len(l) >= 4:
        rows.append({"k3": k, "k3_dd": maxdd(l.shipment), "k3_dds": maxdd(l.ship_share),
                     "k3_ddv": maxdd(l.value_added)})
K3 = pd.DataFrame(rows)
KD = D.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                         cn_growth_e=("cn_growth_e", "mean"),
                         mid=("mid", "first")).reset_index().merge(K3, on="k3", how="inner")
log(f"  k3 표본 {len(KD)}개")
log(f"\n  [기존 DV 대비 상관 — 같은 해상도끼리]")
for c in ("k3_dd", "k3_dds", "k3_ddv"):
    s = KD.dropna(subset=[c])
    log(f"    corr(기존 dd_late, {c:7s}) = {s.dd_late.corr(s[c]):+.3f}  "
        f"(Spearman {s.dd_late.corr(s[c], method='spearman'):+.3f}, n={len(s)})")

log(f"\n  [PADI 예측력 — k3 해상도에서]")
b, p, r2, n = fit(KD.dropna(subset=["dd_late"]), "dd_late")
log(f"    기존 DV     β={b:+8.2f} p={p:.4f} R²={r2:.3f} n={n}   ← 원래 결과")
for c in ("k3_dd", "k3_dds", "k3_ddv"):
    s = KD.dropna(subset=[c])
    b, p, r2, n = fit(s, c)
    log(f"    {c:11s} β={b:+8.2f} p={p:.4f} R²={r2:.3f} n={n}")

log("\n  → 신규 DV 를 같은 해상도(k3)로 올려도 신호가 안 살아나면 원인은 해상도가 아니라")
log("    측정(H1) 또는 빈도(H2) 다. 살아나면 원인은 해상도(H3) 다.")

# ══════════════════════════════════════════════════════════════
log("\n[H2 검정] 기존 DV 를 연 단위로 다운샘플 — 같은 빈도로 맞춘다")
src = os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표", "kosis_광공업생산지수_월별.csv")
P = pd.read_csv(src, encoding="utf-8-sig", dtype=str)
P.columns = [c.strip() for c in P.columns]
P["code"] = P["산업코드"].astype(str).str.strip()
P["yr"] = P["연월"].astype(str).str[:4].astype(int)
P["v"] = pd.to_numeric(P["값"], errors="coerce")
P = P[P.code.str.match(r"^C\d{3}$", na=False)].copy()
P["k3"] = P.code.str[1:4]
itm = P["항목명"].value_counts().index[0]
P = P[P["항목명"] == itm]
log(f"  사용 항목: {itm} | 소분류 {P.k3.nunique()}개")
ann = P.groupby(["k3", "yr"], as_index=False).v.mean()     # 연평균
rows = []
for k, g in ann.groupby("k3"):
    g = g.sort_values("yr")
    l = g[(g.yr >= 2017) & (g.yr <= 2024)]
    e = g[(g.yr >= 2008) & (g.yr < 2017)]
    if len(l) >= 4 and len(e) >= 4:
        rows.append({"k3": k, "dd_ann_late": maxdd(l.v)})
AN = pd.DataFrame(rows)
KA = KD.merge(AN, on="k3", how="left")
s = KA.dropna(subset=["dd_ann_late"])
log(f"  연 단위 다운샘플 표본 {len(s)}개")
log(f"    corr(월별 dd_late, 연별 dd_ann_late) = {s.dd_late.corr(s.dd_ann_late):+.3f}")
b, p, r2, n = fit(s, "dd_ann_late")
log(f"    PADI 예측력  β={b:+8.2f} p={p:.4f} R²={r2:.3f} n={n}")
b, p, r2, n = fit(s, "dd_late")
log(f"    (대조) 월별  β={b:+8.2f} p={p:.4f} R²={r2:.3f} n={n}")
log("  → 연 단위로 낮췄을 때 신호가 죽으면 원인은 빈도(H2) 다.")

# ══════════════════════════════════════════════════════════════
log("\n[보조] 낙폭 크기 비교 — 연별이 낙폭을 얼마나 놓치는가")
if len(s):
    log(f"  월별 낙폭 평균 {s.dd_late.mean():+.2f}%  연별 낙폭 평균 {s.dd_ann_late.mean():+.2f}%")
    log(f"  출하액(연별) 낙폭 평균 {KD.k3_dd.mean():+.2f}%")

KD.to_csv(os.path.join(FUS, "diag_k3_dv.csv"), index=False, encoding="utf-8-sig")
KA.to_csv(os.path.join(FUS, "diag_k3_dv_ann.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {FUS}")

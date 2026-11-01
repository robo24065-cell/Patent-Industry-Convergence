# -*- coding: utf-8 -*-
"""
명목/실질을 가른다 — 동일 개념(출하)의 실질 월별 지수 vs 명목 연별 조사액
────────────────────────────────────────────────────────────────
앞선 진단에서 신규 5자리 DV(광업제조업조사 명목 출하액, 연별)로는 신호가 사라졌고,
원인 후보가 [측정] [빈도] [명목/실질] 로 남았다.

KOSIS 광공업동태조사에는 생산지수 외에 **출하지수·재고지수**가 함께 있다(3자리, 월별, 실질).
따라서 '출하'라는 **동일 개념**을 실질·월별로 잰 계열이 존재한다.

  광업제조업조사 출하액  : 출하 · 명목 · 연별 · 5자리
  KOSIS 출하지수        : 출하 · 실질 · 월별 · 3자리

두 계열을 k3 에서 맞대면 [개념]은 통제되고 [명목/실질 + 빈도]만 남는다.
PADI 가 출하지수 낙폭을 예측하면 → 조사액이 실패한 원인은 명목·연별 때문이다.
예측하지 못하면 → 원래 신호는 '생산' 고유 현상이며 '출하'로는 재현되지 않는다.

덤: 재고지수는 경기 침체의 고전적 신호다. 함께 검정한다.
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
rng  = np.random.default_rng(555)
log  = lambda m: print(m, flush=True)


def maxdd(s):
    s = np.asarray(s, float); s = s[~np.isnan(s)]
    if len(s) < 3 or np.nanmax(s) <= 0: return np.nan
    return float(np.min(s / np.maximum.accumulate(s) - 1.0) * 100)


def maxru(s):
    """최대 상승폭 — 재고는 '쌓이는' 것이 위험 신호이므로 반대 방향을 본다."""
    s = np.asarray(s, float); s = s[~np.isnan(s)]
    if len(s) < 3 or np.nanmin(s) <= 0: return np.nan
    return float(np.max(s / np.minimum.accumulate(s) - 1.0) * 100)


log("[1] 광공업동태조사 3개 지수 적재 (월별, 소분류 3자리, 실질)")
P = pd.read_csv(os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표",
                             "kosis_광공업생산지수_월별.csv"), encoding="utf-8-sig", dtype=str)
P.columns = [c.strip() for c in P.columns]
P["code"] = P["산업코드"].astype(str).str.strip()
P["yr"] = P["연월"].astype(str).str[:4].astype(int)
P["v"] = pd.to_numeric(P["값"], errors="coerce")
P = P[P.code.str.match(r"^C\d{3}$", na=False)].copy()
P["k3"] = P.code.str[1:4]
log(f"  소분류 {P.k3.nunique()}개 | 항목 {list(P['항목명'].unique())}")

rows = []
for (k, item), g in P.groupby(["k3", "항목명"]):
    g = g.sort_values("연월")
    l = g[(g.yr >= 2017) & (g.yr <= 2024)]
    if len(l) < 24:
        continue
    tag = {"생산지수(원지수)": "prod", "출하지수(원지수)": "ship", "재고지수(원지수)": "inv"}[item]
    rows.append({"k3": k, "kind": tag, "dd": maxdd(l.v), "ru": maxru(l.v)})
X = pd.DataFrame(rows).pivot(index="k3", columns="kind", values="dd").reset_index()
X.columns.name = None
X = X.rename(columns={"prod": "dd_prod", "ship": "dd_ship", "inv": "dd_inv"})
IV = pd.DataFrame(rows).pivot(index="k3", columns="kind", values="ru").reset_index()
X["ru_inv"] = X.k3.map(dict(zip(IV.k3, IV["inv"])))
log(f"  낙폭 산출 {len(X)}개 소분류")

log("\n[2] 결합")
KD = pd.read_csv(os.path.join(FUS, "diag_k3_dv.csv"), encoding="utf-8-sig", dtype={"k3": str, "mid": str})
D = KD.merge(X, on="k3", how="left")
log(f"  표본 {len(D)} | 결측 dd_ship {D.dd_ship.isna().sum()}")

log("\n[3] ★ 계열 간 상관 — 무엇이 무엇과 같은 현상인가")
cols = ["dd_late", "dd_prod", "dd_ship", "dd_inv", "k3_dd", "k3_dds", "k3_ddv"]
lab = {"dd_late": "기존DV(생산,월,실질)", "dd_prod": "생산지수(월,실질)",
       "dd_ship": "출하지수(월,실질)", "dd_inv": "재고지수(월,실질)",
       "k3_dd": "조사출하액(연,명목)", "k3_dds": "조사출하점유율(연)",
       "k3_ddv": "조사부가가치(연,명목)"}
C = D[cols].corr()
log("  " + " " * 22 + "".join(f"{lab[c][:9]:>11s}" for c in cols))
for r in cols:
    log(f"  {lab[r]:22s}" + "".join(f"{C.loc[r,c]:+11.3f}" for c in cols))

log("\n[4] ★ 결정적 — PADI 가 각 낙폭을 예측하는가 (k3, n=23)")


def fit(d, y):
    s = d.dropna(subset=[y, "padi_e", "cn_growth_e"])
    if len(s) < 10: return None
    f = smf.ols(f"{y} ~ padi_e + cn_growth_e", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.mid})
    return f.params.padi_e, f.pvalues.padi_e, f.rsquared, len(s)


for c in cols:
    r = fit(D, c)
    if r:
        b, p, r2, n = r
        log(f"  {lab[c]:22s} β={b:+8.2f} p={p:.4f} R²={r2:.3f} n={n} "
            f"{'유의' if p < .05 else ''}")
r = fit(D, "ru_inv")
if r:
    b, p, r2, n = r
    log(f"  {'재고 최대증가(월,실질)':22s} β={b:+8.2f} p={p:.4f} R²={r2:.3f} n={n} "
        f"{'유의' if p < .05 else ''}")

log("\n[5] 판정")
rs, rp = fit(D, "dd_ship"), fit(D, "dd_prod")
if rs and rp:
    log(f"  생산지수 낙폭 예측: R²={rp[2]:.3f} p={rp[1]:.4f}")
    log(f"  출하지수 낙폭 예측: R²={rs[2]:.3f} p={rs[1]:.4f}")
    if rs[1] < .05:
        log("  → 실질·월별 '출하'로도 신호가 재현된다.")
        log("    조사 출하액이 실패한 원인은 개념이 아니라 **명목·연별** 때문이다.")
    else:
        log("  → 실질·월별 출하로도 신호가 재현되지 않는다.")
        log("    원래 신호는 '생산' 계열에 고유한 것이며, 이는 심각한 취약점이다.")

D.to_csv(os.path.join(FUS, "diag_series_compare.csv"), index=False, encoding="utf-8-sig")
log(f"\n저장 → {FUS}/diag_series_compare.csv")

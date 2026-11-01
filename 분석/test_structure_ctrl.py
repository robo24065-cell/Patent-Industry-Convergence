# -*- coding: utf-8 -*-
"""
산업구조 통제변수 검정 — "PADI 는 그냥 영세·저자본 산업 지목 아니냐" 반론 대응
  [0] 해상도 진단: 각 str_* 가 중분류(mid) 내부에서 실제로 변동하는가 (ICC/within-R²)
  [1] PADI 와의 상관 — 구조속성으로 환원되는가
  [2] 기준식 + 통제변수 1개씩 (n=39, n=23 양쪽)
  [3] 통제변수 전부 동시 투입
  [4] 잔차화 검정 — PADI 를 구조변수에 회귀시킨 잔차만으로도 예측되는가
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

STRS = ["str_capital", "str_capital_pw", "str_va_ratio", "str_estab_size",
        "str_labor_prod", "str_wage_share"]
LOGS = {"str_capital_pw": "ln_str_capital_pw", "str_estab_size": "ln_str_estab_size",
        "str_labor_prod": "ln_str_labor_prod", "str_capital": "ln_str_capital"}

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
C = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
D = P.merge(C[["ksic"] + STRS + list(LOGS.values())], on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "cn_growth_e"])
log(f"[표본] n={len(D)}, 구조변수 결측 없음={D[STRS].notna().all().all()}")

# 사용 변수(로그화가 자연스러운 것은 로그판 사용)
USE = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size", "ln_str_labor_prod",
       "str_wage_share", "ln_str_capital_pw"]

# ── [0] 해상도 진단 ─────────────────────────────────────────────
log("\n[0] 해상도 진단 — 중분류(mid, 13개) 내부 변동이 남아있는가")
log("    ICC = 집단간분산/총분산. 1에 가까우면 사실상 2자리 해상도 → 클러스터추론에서 무력")
for v in USE:
    g = D.groupby("mid")[v]
    ssb = ((g.transform("mean") - D[v].mean()) ** 2).sum()
    sst = ((D[v] - D[v].mean()) ** 2).sum()
    icc = ssb / sst
    # k3 내부 변동
    g3 = D.groupby("k3")[v]
    ssb3 = ((g3.transform("mean") - D[v].mean()) ** 2).sum()
    log(f"   {v:22s} ICC(mid)={icc:.3f}  within-mid 변동={1-icc:6.1%} | ICC(k3)={ssb3/sst:.3f}")

# ── [1] PADI 가 구조속성으로 환원되는가 ─────────────────────────
log("\n[1] corr(PADI_early, 구조변수) — |r| 이 크면 '영세산업 지목' 반론이 성립")
for v in USE:
    r = D.padi_e.corr(D[v]); rs = D.padi_e.corr(D[v], method="spearman")
    log(f"   {v:22s} r={r:+.3f}  rho={rs:+.3f}")
fR = smf.ols("padi_e ~ " + " + ".join(USE), data=D).fit()
log(f"   PADI ~ 구조변수 전부:  R²={fR.rsquared:.3f} adj={fR.rsquared_adj:.3f} "
    f"F-p={fR.f_pvalue:.4f}  → PADI 분산의 {fR.rsquared:.1%} 만 구조속성으로 설명")

log("\n[1b] corr(dd_late, 구조변수) — 구조변수 자체가 침체를 예측하는가")
for v in USE:
    log(f"   {v:22s} r={D.dd_late.corr(D[v]):+.3f}  rho={D.dd_late.corr(D[v], method='spearman'):+.3f}")


# ── 공통: 두 표본(n=39, n=23) ──────────────────────────────────
def make_k3(d, extra):
    a = d.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                            cn_growth_e=("cn_growth_e", "mean"), mid=("mid", "first")).reset_index()
    ex = d.groupby("k3")[extra].mean().reset_index()
    return a.merge(ex, on="k3")


A = make_k3(D, USE)
SAMPLES = [("n=39 원자료", D), ("n=23 k3집계", A)]


def fit(d, formula):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.mid})


log("\n[2] 기준식 + 통제변수 1개씩")
for nm, d in SAMPLES:
    base = fit(d, "dd_late ~ padi_e + cn_growth_e")
    log(f"\n  ── {nm} (유효클러스터 mid={d.mid.nunique()}개)")
    log(f"   {'기준식':26s} β_PADI={base.params.padi_e:+8.2f} "
        f"z={base.tvalues.padi_e:+6.2f} p={base.pvalues.padi_e:.4f} R²={base.rsquared:.3f}")
    for v in USE:
        f = fit(d, f"dd_late ~ padi_e + cn_growth_e + {v}")
        chg = (f.params.padi_e - base.params.padi_e) / abs(base.params.padi_e) * 100
        log(f"   +{v:25s} β_PADI={f.params.padi_e:+8.2f} "
            f"z={f.tvalues.padi_e:+6.2f} p={f.pvalues.padi_e:.4f} R²={f.rsquared:.3f} "
            f"| Δβ={chg:+6.1f}% | β_ctrl={f.params[v]:+8.3f} p={f.pvalues[v]:.3f}")

log("\n[3] 통제변수 동시 투입 (과적합 위험 — 참고용)")
for nm, d in SAMPLES:
    base = fit(d, "dd_late ~ padi_e + cn_growth_e")
    # 다중공선 회피: 자본집약도 2종 중 출하액 대비만, 노동생산성/부가가치율 동시 포함
    S1 = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size"]
    S2 = USE
    for tag, S in (("핵심3", S1), ("전체6", S2)):
        f = fit(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        chg = (f.params.padi_e - base.params.padi_e) / abs(base.params.padi_e) * 100
        log(f"   {nm:12s} {tag}: β_PADI={f.params.padi_e:+8.2f} z={f.tvalues.padi_e:+6.2f} "
            f"p={f.pvalues.padi_e:.4f} R²={f.rsquared:.3f} adj={f.rsquared_adj:.3f} Δβ={chg:+6.1f}%")

log("\n[4] 잔차화 검정 — PADI 에서 구조속성 성분을 제거한 나머지만으로 예측")
for nm, d in SAMPLES:
    d = d.copy()
    d["padi_resid"] = smf.ols("padi_e ~ " + " + ".join(USE), data=d).fit().resid
    f = fit(d, "dd_late ~ padi_resid + cn_growth_e")
    log(f"   {nm:12s} β_resid={f.params.padi_resid:+8.2f} z={f.tvalues.padi_resid:+6.2f} "
        f"p={f.pvalues.padi_resid:.4f} R²={f.rsquared:.3f}")

D.to_csv(os.path.join(FUS, "with_structure.csv"), index=False, encoding="utf-8-sig")
log(f"\n[저장] {os.path.join(FUS, 'with_structure.csv')}")

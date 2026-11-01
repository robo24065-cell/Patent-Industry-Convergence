# -*- coding: utf-8 -*-
"""
산업구조 환원 정면 돌파 — 산업 내 층간 격차(within-industry stratum gap)
────────────────────────────────────────────────────────────────
[문제] PADI 수준은 산업 구조와 강하게 겹친다.
       corr(PADI, 노동생산성)=-0.65, corr(PADI, 임금몫)=+0.62 (n=23)
       구조 6지표를 통제하면 β_PADI 가 5% 밖으로 밀린다.
       → "PADI 는 그냥 노동집약·저생산성 산업을 지목하는 것" 이라는 반론을 차단하지 못했다.

[아이디어] 산업 구조 속성은 **산업 단위** 값이다. 같은 산업 안의 두 층(저가치층 vs 전체)은
       동일한 자본집약도·생산성·임금몫·사업체규모를 공유한다.
       따라서 두 층의 PADI 차이를 취하면 **산업 수준 구조 속성이 차분으로 소거된다.**

           gap_j = PADI_저가치층(j) − PADI_전체(j)

       이 격차가 여전히 후반기 낙폭을 예측한다면, 그 예측력은 산업 구조로 환원될 수 없다.
       구조가 설명할 수 있는 것은 두 층에 공통으로 작용하기 때문이다.

[사전 등록] 방향 예측 없음. 격차가 예측하지 못하면 그대로 보고한다.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
rng  = np.random.default_rng(202)
log  = lambda m: print(m, flush=True)
out  = {}

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
S = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig", dtype={"ksic": str})
V = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
     "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"]
V = [v for v in V if v in S.columns]
D = P.merge(S[["ksic"] + V], on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "padi_lo_e", "cn_growth_e"] + V)
D["gap"] = D.padi_lo_e - D.padi_e
log(f"표본 {len(D)}산업 | 클러스터 {D.mid.nunique()}개")
log(f"  PADI_전체   평균 {D.padi_e.mean():.3f}  SD {D.padi_e.std():.3f}")
log(f"  PADI_저가치 평균 {D.padi_lo_e.mean():.3f}  SD {D.padi_lo_e.std():.3f}")
log(f"  격차 gap    평균 {D.gap.mean():+.3f}  SD {D.gap.std():.3f}  "
    f"범위 [{D.gap.min():+.3f}, {D.gap.max():+.3f}]")


def wild_boot(df, formula, var, B=2999):
    f = smf.ols(formula, data=df).fit()
    t_obs = f.params[var] / f.bse[var]
    lhs = formula.split("~")[0].strip()
    terms = [t.strip() for t in formula.split("~")[1].split("+") if t.strip() != var]
    f0 = smf.ols(f"{lhs} ~ " + ("+".join(terms) if terms else "1"), data=df).fit()
    r0, fit0 = f0.resid.values, f0.fittedvalues.values
    g = df["mid"].values; uq = np.unique(g); ts = []
    for _ in range(B):
        wm = dict(zip(uq, rng.choice([-1.0, 1.0], size=len(uq))))
        d2 = df.copy(); d2[lhs] = fit0 + r0 * np.array([wm[x] for x in g])
        try:
            fb = smf.ols(formula, data=d2).fit()
            ts.append(fb.params[var] / fb.bse[var])
        except Exception:
            pass
    return t_obs, float((np.abs(np.array(ts)) >= abs(t_obs)).mean())


def run(df, xvar, ctrls, tag):
    fml = f"dd_late ~ {xvar} + cn_growth_e" + "".join(f" + {c}" for c in ctrls)
    f = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.mid})
    t, wp = wild_boot(df, fml, xvar)
    log(f"  {tag:34s} n={len(df):3d} β={f.params[xvar]:+8.2f} R²={f.rsquared:.3f} "
        f"wild-p={wp:.4f} {'유의' if wp < .05 else '비유의'}")
    return {"tag": tag, "x": xvar, "n": len(df), "beta": float(f.params[xvar]),
            "r2": float(f.rsquared), "wild_p": wp}


# ══════════════════════════════════════════════════════════════
log("\n[1] ★ 핵심 — 격차는 산업 구조와 얼마나 겹치는가")
log("  PADI 수준이 구조와 강하게 상관했다면, 격차는 그 상관이 사라져야 한다.")
log(f"  {'구조지표':22s} {'corr(구조, PADI수준)':>20s} {'corr(구조, 격차)':>18s}")
rows = []
for v in V:
    r_lvl, r_gap = D[v].corr(D.padi_e), D[v].corr(D.gap)
    log(f"  {v:22s} {r_lvl:+20.3f} {r_gap:+18.3f}")
    rows.append({"var": v, "r_level": r_lvl, "r_gap": r_gap})
C = pd.DataFrame(rows)
log(f"\n  평균 |상관|:  수준 {C.r_level.abs().mean():.3f}  →  격차 {C.r_gap.abs().mean():.3f}")
red = 1 - C.r_gap.abs().mean() / max(C.r_level.abs().mean(), 1e-9)
log(f"  구조와의 겹침이 {red*100:.0f}% 감소")
# 구조가 각각을 얼마나 설명하는가
ctl = "+".join(V)
m_lvl = smf.ols(f"padi_e ~ {ctl}", data=D).fit()
m_gap = smf.ols(f"gap ~ {ctl}", data=D).fit()
log(f"  구조6 → PADI수준 설명: R²={m_lvl.rsquared:.3f} F={m_lvl.fvalue:.2f} p={m_lvl.f_pvalue:.4f}")
log(f"  구조6 → 격차   설명: R²={m_gap.rsquared:.3f} F={m_gap.fvalue:.2f} p={m_gap.f_pvalue:.4f}")
out["overlap"] = {"r_level_mean": float(C.r_level.abs().mean()),
                  "r_gap_mean": float(C.r_gap.abs().mean()),
                  "R2_struct_level": float(m_lvl.rsquared), "p_struct_level": float(m_lvl.f_pvalue),
                  "R2_struct_gap": float(m_gap.rsquared), "p_struct_gap": float(m_gap.f_pvalue)}

# ══════════════════════════════════════════════════════════════
log("\n[2] 격차가 후반기 낙폭을 예측하는가")
res = []
res.append(run(D, "padi_e", [], "PADI 수준 (기준)"))
res.append(run(D, "gap", [], "★ 층간 격차"))
res.append(run(D, "padi_lo_e", [], "PADI 저가치층 수준"))

log("\n[3] ★ 결정적 — 격차에 구조 6지표를 전부 통제")
res.append(run(D, "padi_e", V, "PADI 수준 + 구조6"))
res.append(run(D, "gap", V, "★ 층간 격차 + 구조6"))

log("\n[4] 격차와 수준을 동시 투입 — 어느 쪽이 고유 정보인가")
f = smf.ols("dd_late ~ padi_e + gap + cn_growth_e", data=D).fit(
    cov_type="cluster", cov_kwds={"groups": D.mid})
log(f"  β_수준={f.params.padi_e:+.2f} (p={f.pvalues.padi_e:.4f})  "
    f"β_격차={f.params.gap:+.2f} (p={f.pvalues.gap:.4f})  R²={f.rsquared:.3f}")
log(f"  corr(수준, 격차) = {D.padi_e.corr(D.gap):+.3f}")

# ══════════════════════════════════════════════════════════════
log("\n[5] k3 집계 실질표본 n=23")
K = D.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                        padi_lo_e=("padi_lo_e", "mean"), gap=("gap", "mean"),
                        cn_growth_e=("cn_growth_e", "mean"), mid=("mid", "first"),
                        **{v: (v, "mean") for v in V}).reset_index()
res.append(run(K, "padi_e", [], "PADI 수준 (n=23)"))
res.append(run(K, "gap", [], "★ 층간 격차 (n=23)"))
res.append(run(K, "padi_e", V, "PADI 수준 + 구조6 (n=23)"))
res.append(run(K, "gap", V, "★ 층간 격차 + 구조6 (n=23)"))

R = pd.DataFrame(res)
R.to_csv(os.path.join(FUS, "structure_gap_results.csv"), index=False, encoding="utf-8-sig")
C.to_csv(os.path.join(FUS, "structure_gap_overlap.csv"), index=False, encoding="utf-8-sig")
out["specs"] = res
with open(os.path.join(FUS, "structure_gap_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

log("\n[판정]")
g39 = [r for r in res if r["tag"].startswith("★ 층간 격차 + 구조6") and r["n"] == len(D)]
g23 = [r for r in res if "격차 + 구조6 (n=23)" in r["tag"]]
if g39 and g23:
    ok = g39[0]["wild_p"] < .05 and g23[0]["wild_p"] < .05
    log("  " + ("★ 격차가 구조 통제 후에도 양 표본에서 살아남았다 → 구조 환원 차단 성공"
                if ok else
                "격차도 구조 통제를 견디지 못했다 → 구조 환원을 차단하지 못했다. 정직하게 보고할 것."))
log(f"\n저장 → {FUS}")

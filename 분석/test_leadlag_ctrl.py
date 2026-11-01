# -*- coding: utf-8 -*-
"""
전반기 대중 수입침투(5자리 해상도) 를 기준모형에 통제변수로 투입했을 때
β_PADI 가 살아남는지 검정.

기준식: dd_late ~ padi_e + cn_growth_e            (β_padi=-38.70, R²=0.333, n=39)
추정:   smf.ols(...).fit(cov_type='cluster', cov_kwds={'groups': mid})

두 표본 모두 보고: (a) n=39 원자료, (b) k3 집계 n=23.
β_PADI 가 죽으면 죽었다고 그대로 보고한다.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
F = os.path.join(BASE, "분석", "fusion")
log = lambda m: print(m, flush=True)

d = pd.read_csv(os.path.join(F, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
c = pd.read_csv(os.path.join(F, "ctrl_leadlag_early.csv"), encoding="utf-8-sig",
                dtype={"ksic": str})
d = d.merge(c.drop(columns=[x for x in ("mid",) if x in c.columns]),
            on="ksic", how="left")

CTRLS = ["cnimp_share_e", "cnimp_rise_e", "cnimp_lnusd_e", "cnimp_growth_e"]


def fit(df, formula, grp, tag):
    vs = [v.strip() for v in formula.replace("~", "+").split("+")]
    dd = df.dropna(subset=[v for v in vs if v in df.columns] + [grp])
    try:
        m = smf.ols(formula, data=dd).fit(cov_type="cluster",
                                          cov_kwds={"groups": dd[grp]})
    except Exception as e:
        log(f"  {tag}: 실패 {e}"); return
    b = m.params.get("padi_e", np.nan); t = m.tvalues.get("padi_e", np.nan)
    p = m.pvalues.get("padi_e", np.nan)
    log(f"  {tag}")
    log(f"    β_padi={b:+7.2f}  t={t:+5.2f}  p={p:.4f}  "
        f"R²={m.rsquared:.3f}  n={int(m.nobs)}  클러스터={dd[grp].nunique()}")
    for v in CTRLS:
        if v in m.params:
            log(f"      {v:16s} β={m.params[v]:+8.3f} t={m.tvalues[v]:+5.2f} "
                f"p={m.pvalues[v]:.3f}")


log("=" * 72)
log("[A] n=39 원자료 (클러스터: mid)")
fit(d, "dd_late ~ padi_e + cn_growth_e", "mid", "기준식 (재현)")
for v in CTRLS:
    fit(d, f"dd_late ~ padi_e + cn_growth_e + {v}", "mid", f"+ {v}")
fit(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(CTRLS), "mid", "+ 전체 동시투입")

log("")
log("=" * 72)
log("[B] k3 집계 n=23 (클러스터: mid)")
agg = {c_: "mean" for c_ in ["dd_late", "padi_e", "cn_growth_e"] + CTRLS}
k = d.groupby("k3", as_index=False).agg(agg)
k["mid"] = k.k3.str[:2]
log(f"  k3 단위 {len(k)}개")
fit(k, "dd_late ~ padi_e + cn_growth_e", "mid", "기준식 (재현)")
for v in CTRLS:
    fit(k, f"dd_late ~ padi_e + cn_growth_e + {v}", "mid", f"+ {v}")
fit(k, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(CTRLS), "mid", "+ 전체 동시투입")

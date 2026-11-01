# -*- coding: utf-8 -*-
"""
융합데이터 횡단면 모형의 강건성 — n=39, 예측변수 3개
  최대낙폭 ~ PADI + 중국비중 + 중국증가율  (R²=0.429, 전 변수 유의)
검정: 클러스터 SE / LOO / VIF / 대안 종속변수 / 층별(hi·lo)
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
rng = np.random.default_rng(5)
log = lambda m: print(m, flush=True)

a = pd.read_csv(os.path.join(FUS, "fusion_agg.csv"), encoding="utf-8-sig", dtype={"ksic": str, "mid": str})
a = a.dropna(subset=["maxdd", "padi_all", "cn_share", "cn_growth"])
log(f"n={len(a)}, 중분류 {a.mid.nunique()}개")

F = "maxdd ~ padi_all + cn_share + cn_growth"
log("\n[1] 표준오차 3종 비교")
f_ols = smf.ols(F, data=a).fit()
f_hc = smf.ols(F, data=a).fit(cov_type="HC3")
f_cl = smf.ols(F, data=a).fit(cov_type="cluster", cov_kwds={"groups": a.mid})
tab = pd.DataFrame({
    "β": f_ols.params, "t_OLS": f_ols.tvalues, "t_HC3": f_hc.tvalues,
    "t_클러스터": f_cl.tvalues, "p_클러스터": f_cl.pvalues}).drop("Intercept")
log(tab.round(3).to_string())
log(f"  R²={f_ols.rsquared:.3f}  adj.R²={f_ols.rsquared_adj:.3f}  F p={f_ols.f_pvalue:.4f}")

log("\n[2] 다중공선성 (VIF)")
X = a[["padi_all", "cn_share", "cn_growth"]]
for c in X.columns:
    others = [x for x in X.columns if x != c]
    r2 = smf.ols(f"{c} ~ " + " + ".join(others), data=a).fit().rsquared
    log(f"  {c:12s} VIF = {1/(1-r2):.2f}")

log("\n[3] Leave-one-industry-out (계수 안정성)")
loo = []
for i in range(len(a)):
    s = a.drop(a.index[i])
    ff = smf.ols(F, data=s).fit()
    loo.append({"dropped": a.ksic.iloc[i], "b_padi": ff.params["padi_all"],
                "b_cn": ff.params["cn_share"], "R2": ff.rsquared})
L = pd.DataFrame(loo)
log(f"  β_PADI: 전체 {f_ols.params['padi_all']:+.1f} | LOO 범위 [{L.b_padi.min():+.1f}, {L.b_padi.max():+.1f}] SD {L.b_padi.std():.2f}")
log(f"  R²    : 전체 {f_ols.rsquared:.3f} | LOO 범위 [{L.R2.min():.3f}, {L.R2.max():.3f}]")
worst = L.nlargest(1, "b_padi").iloc[0]
log(f"  최대 영향: {worst.dropped} 제거 시 β={worst.b_padi:+.1f}")

log("\n[4] Leave-one-중분류-out")
for mid in sorted(a.mid.unique()):
    s = a[a.mid != mid]
    if len(s) < 12: continue
    ff = smf.ols(F, data=s).fit()
    flag = " ←부호변화" if ff.params["padi_all"] > 0 else ""
    log(f"  {mid} 제외(n={len(s)}): β_PADI={ff.params['padi_all']:+.1f} t={ff.tvalues['padi_all']:+.2f} R²={ff.rsquared:.3f}{flag}")

log("\n[5] 층별 모형 (가치 층화)")
for tag in ("padi_all", "padi_hi", "padi_lo"):
    d = a.dropna(subset=[tag])
    ff = smf.ols(f"maxdd ~ {tag} + cn_share + cn_growth", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.mid})
    log(f"  {tag:9s}: β={ff.params[tag]:+7.1f} t={ff.tvalues[tag]:+5.2f} p={ff.pvalues[tag]:.4f} R²={ff.rsquared:.3f}")

log("\n[6] 대안 종속변수")
alt = pd.read_csv(os.path.join(BASE, "분석", "stress", "severity_agg.csv"),
                  encoding="utf-8-sig", dtype={"unit": str}).rename(columns={"unit": "ksic"})
m = a.merge(alt[["ksic", "sev_mean_neg", "sev_p05", "downside_sd", "freq_neg"]], on="ksic", how="left")
for y in ("sev_mean_neg", "sev_p05", "downside_sd", "freq_neg"):
    d = m.dropna(subset=[y])
    ff = smf.ols(f"{y} ~ padi_all + cn_share + cn_growth", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.mid})
    log(f"  {y:14s}: β_PADI={ff.params['padi_all']:+7.2f} t={ff.tvalues['padi_all']:+5.2f} "
        f"p={ff.pvalues['padi_all']:.4f} R²={ff.rsquared:.3f}")

log("\n[7] 부트스트랩 계수 CI (중분류 클러스터, 3000회)")
uq = a.mid.unique()
bs = []
for _ in range(3000):
    pick = rng.choice(uq, len(uq), replace=True)
    idx = np.concatenate([np.where(a.mid.values == c)[0] for c in pick])
    s = a.iloc[idx]
    if s.ksic.nunique() < 10: continue
    try:
        ff = smf.ols(F, data=s).fit()
        bs.append([ff.params["padi_all"], ff.params["cn_share"], ff.params["cn_growth"], ff.rsquared])
    except Exception:
        pass
B = pd.DataFrame(bs, columns=["padi", "cn_share", "cn_growth", "R2"])
for c in ("padi", "cn_share", "cn_growth"):
    lo, hi = np.percentile(B[c], [2.5, 97.5])
    sig = "유의" if (lo > 0 or hi < 0) else "비유의"
    log(f"  {c:10s} CI[{lo:+.2f}, {hi:+.2f}] {sig}")
log(f"  R² 중앙값 {B.R2.median():.3f}, CI[{np.percentile(B.R2,2.5):.3f}, {np.percentile(B.R2,97.5):.3f}]")

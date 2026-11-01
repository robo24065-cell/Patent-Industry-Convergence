# -*- coding: utf-8 -*-
"""
실질 표본 확인 — 종속변수가 KSIC 3자리라 n=39는 명목, 실질은 23
  [1] k3 집계 분석 (n=23) — 설명변수를 위험집합 가중평균으로 집계
  [2] 39 vs 23 결과 비교
  [3] wild cluster bootstrap-t (G=13 소표본 클러스터 대응)
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
rng = np.random.default_rng(53)
log = lambda m: print(m, flush=True)

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
P = P.dropna(subset=["dd_late", "padi_e", "cn_growth_e"])

log("[1] 명목 n=39 (기존)")
f39 = smf.ols("dd_late ~ padi_e + cn_growth_e", data=P).fit(
    cov_type="cluster", cov_kwds={"groups": P.mid})
log(f"  β_PADI={f39.params.padi_e:+.2f} z={f39.tvalues.padi_e:+.2f} p={f39.pvalues.padi_e:.4f}")
log(f"  β_CN  ={f39.params.cn_growth_e:+.3f} z={f39.tvalues.cn_growth_e:+.2f} p={f39.pvalues.cn_growth_e:.4f}")
log(f"  R²={f39.rsquared:.3f}")

log("\n[2] 실질 n=23 (k3 집계 — 설명변수 단순평균)")
A = P.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                        cn_growth_e=("cn_growth_e", "mean"), mid=("mid", "first"),
                        n_sub=("ksic", "size")).reset_index()
log(f"  n={len(A)}, 중분류 {A.mid.nunique()}개")
f23 = smf.ols("dd_late ~ padi_e + cn_growth_e", data=A).fit(
    cov_type="cluster", cov_kwds={"groups": A.mid})
log(f"  β_PADI={f23.params.padi_e:+.2f} z={f23.tvalues.padi_e:+.2f} p={f23.pvalues.padi_e:.4f}")
log(f"  β_CN  ={f23.params.cn_growth_e:+.3f} z={f23.tvalues.cn_growth_e:+.2f} p={f23.pvalues.cn_growth_e:.4f}")
log(f"  R²={f23.rsquared:.3f}  adj={f23.rsquared_adj:.3f}")
r = A.padi_e.corr(A.dd_late)
log(f"  단순상관 corr(PADI, 낙폭) = {r:+.3f}, Spearman = {A.padi_e.corr(A.dd_late, method='spearman'):+.3f}")

log("\n[3] wild cluster bootstrap-t (G=13, Rademacher, 9999회)")


def wild_boot(df, formula, var, G, B=9999):
    f = smf.ols(formula, data=df).fit()
    t_obs = f.params[var] / f.bse[var]
    # 제약 모형 (H0: β_var = 0)
    terms = [t for t in formula.split("~")[1].split("+") if var not in t]
    f0 = smf.ols(formula.split("~")[0] + "~" + "+".join(terms), data=df).fit()
    resid0, fitted0 = f0.resid.values, f0.fittedvalues.values
    groups = df[G].values
    uq = np.unique(groups)
    ts = []
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=len(uq))
        wmap = dict(zip(uq, w))
        yb = fitted0 + resid0 * np.array([wmap[g] for g in groups])
        d2 = df.copy(); d2[formula.split("~")[0].strip()] = yb
        try:
            fb = smf.ols(formula, data=d2).fit()
            ts.append(fb.params[var] / fb.bse[var])
        except Exception:
            pass
    ts = np.array(ts)
    p = float((np.abs(ts) >= abs(t_obs)).mean())
    return t_obs, p, len(ts)


for name, d in (("n=39", P), ("n=23", A)):
    for v in ("padi_e", "cn_growth_e"):
        t, p, B = wild_boot(d, "dd_late ~ padi_e + cn_growth_e", v, "mid", 2999)
        log(f"  {name} {v:12s}: t={t:+.2f} wild-p={p:.4f} {'유의' if p<0.05 else '**비유의**'} (B={B})")

log("\n[4] 판정")
log("  - 우리는 이미 중분류(13개) 클러스터 SE를 사용 → k3(23개)보다 보수적")
log("  - 명목 n=39는 회귀 표기용이며, 추론은 클러스터 단위로 이뤄짐")
A.to_csv(os.path.join(FUS, "agg_k3.csv"), index=False, encoding="utf-8-sig")

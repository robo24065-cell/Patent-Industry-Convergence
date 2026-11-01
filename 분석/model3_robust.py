# -*- coding: utf-8 -*-
"""
3축 모형 강건성 — 후반기 낙폭 ~ 전반기 PADI + 중국증가율 + 출원인집중도(HHI)
주의: HHI는 6개 후보 중 사후 선택 → 다중검정 보정과 함께 '후보'로만 보고
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
rng = np.random.default_rng(23)
log = lambda m: print(m, flush=True)

A = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "mid": str, "k3": str})
A = A.dropna(subset=["dd_late", "padi_e", "cn_growth_e", "hhi"])
log(f"n={len(A)}, 중분류 {A.mid.nunique()}")

log("\n[0] 다중검정 위치 — 제3축 후보 6개 중 HHI 선택")
ps = pd.Series({"HHI": 0.012, "1위점유율": 0.039, "한국상대지위": 0.243,
                "실용신안비중": 0.466, "IPC분야수": 0.603, "IPC엔트로피": 0.983}).sort_values()
q = (ps.values * len(ps) / (np.arange(len(ps)) + 1))
q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
for (k, p), qq in zip(ps.items(), q):
    log(f"  {k:12s} p={p:.3f} → BH q={qq:.3f} {'통과' if qq < 0.10 else ''}")

F2 = "dd_late ~ padi_e + cn_growth_e"
F3 = "dd_late ~ padi_e + cn_growth_e + hhi"
log("\n[1] 2축 vs 3축")
for name, F in (("2축", F2), ("3축", F3)):
    f = smf.ols(F, data=A).fit(cov_type="cluster", cov_kwds={"groups": A.mid})
    log(f"  {name}: R²={f.rsquared:.3f} adj={f.rsquared_adj:.3f}")
    log("    " + f.summary2().tables[1].round(3).to_string().replace("\n", "\n    "))

log("\n[2] VIF (독립성)")
X = A[["padi_e", "cn_growth_e", "hhi"]]
for c in X.columns:
    o = [x for x in X.columns if x != c]
    r2 = smf.ols(f"{c} ~ " + " + ".join(o), data=A).fit().rsquared
    log(f"  {c:14s} VIF={1/(1-r2):.2f}")

log("\n[3] Leave-one-중분류-out (3축)")
for mid in sorted(A.mid.unique()):
    s = A[A.mid != mid]
    if len(s) < 12: continue
    f = smf.ols(F3, data=s).fit()
    log(f"  {mid} 제외(n={len(s)}): β_PADI={f.params['padi_e']:+7.1f} "
        f"β_HHI={f.params['hhi']:+8.1f} R²={f.rsquared:.3f}")

log("\n[4] 클러스터 부트스트랩 계수 CI (3000회)")
uq = A.mid.unique(); bs = []
for _ in range(3000):
    pick = rng.choice(uq, len(uq), replace=True)
    i = np.concatenate([np.where(A.mid.values == c)[0] for c in pick])
    s = A.iloc[i]
    if s.ksic.nunique() < 10: continue
    try:
        f = smf.ols(F3, data=s).fit()
        bs.append([f.params["padi_e"], f.params["cn_growth_e"], f.params["hhi"], f.rsquared])
    except Exception:
        pass
B = pd.DataFrame(bs, columns=["padi", "cn_growth", "hhi", "R2"])
for c in ("padi", "cn_growth", "hhi"):
    lo, hi = np.percentile(B[c], [2.5, 97.5])
    log(f"  {c:10s} CI[{lo:+9.2f}, {hi:+9.2f}] {'유의' if (lo>0 or hi<0) else '**비유의**'}")
log(f"  R² 중앙값 {B.R2.median():.3f} CI[{np.percentile(B.R2,2.5):.3f},{np.percentile(B.R2,97.5):.3f}]")

log("\n[5] 표본외 예측 성능 (LOO 교차검증 RMSE)")
for name, F in (("2축", F2), ("3축", F3)):
    errs = []
    for i in range(len(A)):
        tr, te = A.drop(A.index[i]), A.iloc[[i]]
        f = smf.ols(F, data=tr).fit()
        errs.append(float(te.dd_late.iloc[0] - f.predict(te).iloc[0]))
    rmse = np.sqrt(np.mean(np.square(errs)))
    log(f"  {name}: LOO-RMSE = {rmse:.2f}")
sd = A.dd_late.std()
log(f"  (참고) 종속변수 표준편차 = {sd:.2f} → 예측이 평균보다 나으면 RMSE < {sd:.2f}")

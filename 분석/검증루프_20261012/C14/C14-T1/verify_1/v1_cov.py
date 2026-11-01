# -*- coding: utf-8 -*-
r"""verify_1: selftest 캐시(npz)에서 포함률을 독립 재집계 + 몬테카를로 불확실성 + E4 탈락 원인 분해.
lib14 미사용. 쓰기는 verify_1 안에서만."""
import os, glob, json, sys
import numpy as np
from scipy.stats import t as tdist, norm, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "cache")
out = {}
for S in ("S1", "S2"):
    fs = sorted(glob.glob(os.path.join(CACHE, f"st_{S}_*.npz")), key=lambda f: int(os.path.basename(f).split("_")[2]))
    Z = [dict(np.load(f)) for f in fs]
    d = {k: np.concatenate([z[k] for z in Z]) for k in Z[0]}
    n = len(d["pt"])
    truth = d["pt"].mean(); sd = d["pt"].std(ddof=1)
    r = {"files": [os.path.basename(f) for f in fs], "reps": n, "uniq_pt": int(len(np.unique(d["pt"]))), "truth": truth, "true_sd": sd}
    df = 59
    q = tdist.ppf(0.975, df)
    for e in ("E1", "E2", "E3", "E4", "E4b"):
        lo, hi, se = d[f"{e}_lo"], d[f"{e}_hi"], d[f"{e}_se"]
        k = int(((lo <= truth) & (truth <= hi)).sum())
        ci = binomtest(k, n).proportion_ci(0.95)
        rr = {"k": k, "coverage": k / n, "binom95": [ci.low, ci.high],
              "p_two_sided_vs_0.95": binomtest(k, n, 0.95).pvalue,
              "mean_se/sd": float(se.mean() / sd), "cv_se": float(se.std(ddof=1) / se.mean())}
        # 진실을 MC 평균 대신 '각 반복 제외 평균'(leave-one-out)로 바꿔도 같은가
        loo = (truth * n - d["pt"]) / (n - 1)
        rr["coverage_loo_truth"] = float(np.mean((lo <= loo) & (loo <= hi)))
        if e in ("E3", "E4", "E4b"):
            # CI 대칭 재구성 점검
            rr["max|ci-recon|"] = float(max(np.abs(lo - (d["pt"] - q * se)).max(), np.abs(hi - (d["pt"] + q * se)).max()))
            # 오라클: SE를 평균 SE로 고정(잡음 제거)했을 때 포함률 → 편향 기여
            sbar = np.sqrt((se ** 2).mean())
            rr["coverage_fixedSE_rms"] = float(np.mean(np.abs(d["pt"] - truth) <= q * sbar))
            # 이론 포함률(정규 가정, SE 고정): 2Φ(q·rms/sd)−1
            rr["theory_cov_fixedSE"] = float(2 * norm.cdf(q * sbar / sd) - 1)
        r[e] = rr
    # E3 vs E4/E4b 포함 판정 차이(쌍대) — McNemar 정확 검정
    c3 = (d["E3_lo"] <= truth) & (truth <= d["E3_hi"])
    for e in ("E4", "E4b"):
        ce = (d[f"{e}_lo"] <= truth) & (truth <= d[f"{e}_hi"])
        b = int((c3 & ~ce).sum()); c = int((~c3 & ce).sum())
        r[f"mcnemar_E3_vs_{e}"] = {"E3only": b, "Eonly": c, "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}
    r["corr(E4_se,E3_se)"] = float(np.corrcoef(d["E4_se"], d["E3_se"])[0, 1])
    r["E3_fix_rate"] = float(d["E3_fix"].mean()); r["E4_fix_rate"] = float(d["E4_fix"].mean())
    r["vAB_share"] = float(np.mean(d["vAB"] / (d["vA"] + d["vB"])))
    out[S] = r
json.dump(out, open(os.path.join(HERE, "v1_cov.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
for S in out:
    print(S, out[S]["reps"], out[S]["uniq_pt"], round(out[S]["truth"], 6), round(out[S]["true_sd"], 6))
    for e in ("E1", "E2", "E3", "E4", "E4b"):
        v = out[S][e]
        print(" ", e, v["k"], round(v["coverage"], 3), [round(x, 3) for x in v["binom95"]], "p=%.3g" % v["p_two_sided_vs_0.95"],
              "se/sd", round(v["mean_se/sd"], 3), "cv", round(v["cv_se"], 3), "loo", round(v["coverage_loo_truth"], 3),
              {k: round(v[k], 4) for k in ("max|ci-recon|", "coverage_fixedSE_rms", "theory_cov_fixedSE") if k in v})
    print(" ", out[S]["mcnemar_E3_vs_E4"], out[S]["mcnemar_E3_vs_E4b"], "corr", round(out[S]["corr(E4_se,E3_se)"], 3), "fix", out[S]["E3_fix_rate"], out[S]["E4_fix_rate"])

# -*- coding: utf-8 -*-
"""
5자리 종속변수로 전면 재검정
────────────────────────────────────────────────────────────────
[사전 등록 — 결과를 보기 전에 정한다]
  · 종속변수 후보 4종을 **전부** 보고한다. 좋은 것만 고르지 않는다.
      dd5_late   명목 출하액 낙폭        (물가 오염 있음)
      dd5s_late  출하액 점유율 낙폭      (물가 소거, 상대적 쇠퇴)   ← 주력
      dd5v_late  부가가치 낙폭
      dd5vs_late 부가가치 점유율 낙폭
  · 4개를 동시에 보므로 **다중검정 보정(BH-FDR)** 을 적용해 보고한다.
  · 기존 종속변수와의 상관은 +0.27~+0.33 에 불과하다. 이것은 기존 변수의 '개선판'이 아니라
    **다른 변수**다. 따라서 이 검정은 기존 결과의 재현이 아니라 **독립 재검정**이다.
  · 결과가 나쁘면 그대로 쓴다.

[이 검정이 답해야 할 단 하나의 질문]
  종속변수 해상도를 5자리로 올리면 PADI 와 산업구조를 분리할 수 있는가?
  (3자리 종속변수에서는 β가 40% 축소되고 플라시보 대비 p=0.031 로 공유분산이 확인됐다)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
rng  = np.random.default_rng(31337)
log  = lambda m: print(m, flush=True)
out  = {}

M = pd.read_csv(os.path.join(FUS, "predictive_agg_dv5.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
S = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig", dtype={"ksic": str})
V = [v for v in ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
                 "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"] if v in S.columns]
F = pd.read_csv(os.path.join(FUS, "ctrl_financial.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str})
FV = [c for c in F.columns if c.startswith("fin_") and F[c].dtype != object]
D = M.merge(S[["ksic"] + V], on="ksic", how="left").merge(F[["ksic"] + FV], on="ksic", how="left")
DVS = ["dd5_late", "dd5s_late", "dd5v_late", "dd5vs_late"]
D = D.dropna(subset=["padi_e", "cn_growth_e"] + V)
log(f"표본 {len(D)}산업 | 클러스터 mid {D.mid.nunique()} / k3 {D.k3.nunique()}")


def wild_p(df, dv, ctrls, B=2999, seed=None):
    r = np.random.default_rng(seed if seed is not None else rng.integers(1 << 30))
    fml = f"{dv} ~ padi_e + cn_growth_e" + "".join(f" + {c}" for c in ctrls)
    f = smf.ols(fml, data=df).fit()
    t = f.params.padi_e / f.bse.padi_e
    f0 = smf.ols(f"{dv} ~ cn_growth_e" + "".join(f" + {c}" for c in ctrls), data=df).fit()
    r0, fit0 = f0.resid.values, f0.fittedvalues.values
    g = df["mid"].values; uq = np.unique(g); ts = []
    for _ in range(B):
        wm = dict(zip(uq, r.choice([-1.0, 1.0], size=len(uq))))
        d2 = df.copy(); d2[dv] = fit0 + r0 * np.array([wm[x] for x in g])
        try:
            fb = smf.ols(fml, data=d2).fit()
            ts.append(fb.params.padi_e / fb.bse.padi_e)
        except Exception:
            pass
    return f, float((np.abs(np.array(ts)) >= abs(t)).mean())


def bh(ps):
    p = np.asarray(ps, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


# ══════════════════════════════════════════════════════════════
log("\n[1] 기준모형 — 종속변수 4종 전부")
rows = []
for dv in DVS:
    d = D.dropna(subset=[dv])
    f, wp = wild_p(d, dv, [], seed=11)
    rows.append({"dv": dv, "spec": "기준", "n": len(d), "beta": f.params.padi_e,
                 "r2": f.rsquared, "wild_p": wp})
    log(f"  {dv:11s} n={len(d):3d} β_PADI={f.params.padi_e:+8.2f} "
        f"β_CN={f.params.cn_growth_e:+7.3f} R²={f.rsquared:.3f} wild-p={wp:.4f}")
q = bh([r["wild_p"] for r in rows])
for r, qq in zip(rows, q):
    r["bh_q"] = float(qq)
log("  BH-FDR 보정 q값: " + ", ".join(f"{r['dv']}={r['bh_q']:.4f}" for r in rows))

# ══════════════════════════════════════════════════════════════
log("\n[2] ★ 산업구조 6지표 통제 — 3자리에서 실패했던 그 검정")
for dv in DVS:
    d = D.dropna(subset=[dv])
    f0 = smf.ols(f"{dv} ~ padi_e + cn_growth_e", data=d).fit()
    f, wp = wild_p(d, dv, V, seed=12)
    shrink = 1 - abs(f.params.padi_e / f0.params.padi_e)
    rows.append({"dv": dv, "spec": "구조6", "n": len(d), "beta": f.params.padi_e,
                 "r2": f.rsquared, "wild_p": wp, "shrink": shrink})
    log(f"  {dv:11s} β {f0.params.padi_e:+8.2f} → {f.params.padi_e:+8.2f} "
        f"(축소 {shrink*100:+5.1f}%) R²={f.rsquared:.3f} wild-p={wp:.4f} "
        f"{'유의' if wp < .05 else '비유의'}")

log("\n[3] ★ 플라시보 — 구조6이 무작위 통제보다 더 많이 죽이는가")
for dv in DVS:
    d = D.dropna(subset=[dv])
    b0 = smf.ols(f"{dv} ~ padi_e + cn_growth_e", data=d).fit().params.padi_e
    bS = smf.ols(f"{dv} ~ padi_e + cn_growth_e + " + "+".join(V), data=d).fit().params.padi_e
    rS = smf.ols("padi_e ~ " + "+".join(V), data=d).fit().rsquared
    bs = []
    for _ in range(3000):
        d2 = d.copy()
        for j, v in enumerate(V):
            d2[f"z{j}"] = rng.permutation(d[v].values)
        cs = "+".join(f"z{j}" for j in range(len(V)))
        try:
            bs.append(smf.ols(f"{dv} ~ padi_e + cn_growth_e + {cs}", data=d2).fit().params.padi_e)
        except Exception:
            pass
    bs = np.array(bs)
    tail = float((np.abs(bs) <= abs(bS)).mean())
    log(f"  {dv:11s} 관측축소 {(1-abs(bS/b0))*100:+5.1f}%  플라시보 평균축소 "
        f"{(1-np.abs(bs/b0)).mean()*100:+5.1f}%  ★꼬리 p={tail:.4f}  "
        f"{'공유분산 (여전히 분리 실패)' if tail < .10 else '분리 성공 — 구조는 무작위와 다르지 않다'}")
    out[f"placebo_{dv}"] = {"tail": tail, "shrink": float(1 - abs(bS / b0)),
                            "r2_padi_struct": float(rS)}

log("\n[4] ★ 새로 가능해진 검정 — k3 고정효과 (소분류 내 변동만 사용)")
log("  3자리 종속변수에서는 그룹내 분산이 0이라 아예 불가능했던 검정이다.")
log("  k3 FE 는 소분류 수준의 모든 교란(산업구조 포함)을 통째로 제거한다.")
for dv in DVS:
    d = D.dropna(subset=[dv])
    multi = d.groupby("k3").filter(lambda g: len(g) > 1)
    if len(multi) < 8:
        log(f"  {dv:11s} 복수산업 k3 표본 부족 (n={len(multi)})"); continue
    f = smf.ols(f"{dv} ~ padi_e + cn_growth_e + C(k3)", data=multi).fit(
        cov_type="cluster", cov_kwds={"groups": multi.k3})
    log(f"  {dv:11s} within n={len(multi):3d} (k3 {multi.k3.nunique()}개) "
        f"β_PADI={f.params.padi_e:+8.2f} p={f.pvalues.padi_e:.4f} "
        f"{'유의' if f.pvalues.padi_e < .05 else '비유의'}")
    out[f"k3fe_{dv}"] = {"beta": float(f.params.padi_e), "p": float(f.pvalues.padi_e),
                         "n": len(multi)}

log("\n[5] 재무 통제 (3자리에서 통과했던 검정의 재현)")
for dv in DVS:
    d = D.dropna(subset=[dv] + FV[:4])
    if len(d) < 20: continue
    f, wp = wild_p(d, dv, FV[:4], seed=13)
    log(f"  {dv:11s} +재무4 β={f.params.padi_e:+8.2f} wild-p={wp:.4f} "
        f"{'유의' if wp < .05 else '비유의'}")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(FUS, "dv5_test_results.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(FUS, "dv5_test_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")

# -*- coding: utf-8 -*-
"""
'자유도 고갈'인가 '공유분산'인가 — 두 해석을 가르는 결정적 검정
────────────────────────────────────────────────────────────────
[쟁점] n=23 에 구조6을 넣으면 β_PADI 가 −42.8 → −25.5 로 죽는다(wild-p≈0.32~0.34).
  나의 해석  : 구조 블록도 결합 비유의(p=0.844)이고 모형 전체도 비유의(p=0.215)이니
               '어느 쪽도 식별되지 않는' 자유도 고갈이다.
  검증자 해석: 아니다. 공유분산 때문이다. R²(PADI~구조6)=0.502, β가 40% 축소된다.

[가르는 방법 — 플라시보]
  자유도 고갈이 원인이라면, **아무 변수 6개**를 넣어도 비슷하게 죽어야 한다.
  구조6이 무작위 통제보다 **더 많이** 죽인다면 원인은 자유도가 아니라 공유분산이다.

  귀무분포 생성 3가지 (셋 다 자유도 소모는 구조6과 정확히 동일):
    (a) 순수 난수 6개
    (b) 구조6의 각 열을 독립적으로 무작위 순열 (주변분포 보존, PADI와의 관계만 파괴)
    (c) 다른 실제 통제 6개를 무작위 추출 (재무지표 등 — 현실적 상관구조 보존)

[정직 원칙] 관측값이 귀무분포 꼬리에 있으면 내 해석이 틀린 것이다. 그대로 인정한다.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
rng  = np.random.default_rng(4242)
log  = lambda m: print(m, flush=True)
out  = {}

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str}).dropna(
                subset=["dd_late", "padi_e", "cn_growth_e"])
S = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig", dtype={"ksic": str})
V = [v for v in ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
                 "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"] if v in S.columns]
D = P.merge(S[["ksic"] + V], on="ksic", how="left").dropna(subset=V)
K = D.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                        cn_growth_e=("cn_growth_e", "mean"), mid=("mid", "first"),
                        **{v: (v, "mean") for v in V}).reset_index()
log(f"n=39 {len(D)} | n=23 {len(K)} | 구조지표 {len(V)}종")

# ══════════════════════════════════════════════════════════════
log("\n[0] 먼저 확인 — n=39 가 독립 증거인가 (검증자 주장 #4)")
w = D.groupby("k3").dd_late.nunique()
log(f"  k3 내 dd_late 고유값 분포: {w.value_counts().to_dict()}")
within_var = D.groupby("k3").dd_late.var(ddof=0).fillna(0).sum()
log(f"  k3 내 dd_late 총 분산 = {within_var:.10f}")
if within_var < 1e-9:
    log("  ★ 확인됨: 종속변수의 그룹내 분산이 정확히 0.")
    log("    → n=39 회귀는 n=23 을 'KSIC5 개수'로 가중한 것과 동일하다.")
    log("    → 그 가중치는 종속변수와 무관한 데이터 구축 부산물이다.")
    log("    → **n=39 는 n=23 과 독립된 증거가 아니다.** 검증자 주장이 맞다.")
    fw = smf.ols("dd_late ~ padi_e + C(k3)", data=D).fit()
    log(f"    그룹내(k3 고정효과) β_PADI = {fw.params.padi_e:+.4f} (p={fw.pvalues.padi_e:.4f}) — 신호 전무")
out["n39_independent"] = bool(within_var >= 1e-9)


def wild_p(df, ctrls, B=1999, seed=None):
    r = np.random.default_rng(seed if seed is not None else rng.integers(1 << 30))
    fml = "dd_late ~ padi_e + cn_growth_e" + "".join(f" + {c}" for c in ctrls)
    f = smf.ols(fml, data=df).fit()
    t_obs = f.params.padi_e / f.bse.padi_e
    f0 = smf.ols("dd_late ~ cn_growth_e" + "".join(f" + {c}" for c in ctrls), data=df).fit()
    r0, fit0 = f0.resid.values, f0.fittedvalues.values
    g = df["mid"].values; uq = np.unique(g); ts = []
    for _ in range(B):
        wm = dict(zip(uq, r.choice([-1.0, 1.0], size=len(uq))))
        d2 = df.copy(); d2["dd_late"] = fit0 + r0 * np.array([wm[x] for x in g])
        try:
            fb = smf.ols(fml, data=d2).fit()
            ts.append(fb.params.padi_e / fb.bse.padi_e)
        except Exception:
            pass
    return float(f.params.padi_e), float((np.abs(np.array(ts)) >= abs(t_obs)).mean())


# ══════════════════════════════════════════════════════════════
log("\n[1] 관측값 — 구조6 투입 (n=23)")
b_obs, p_obs = wild_p(K, V, B=2999, seed=7)
b_base, p_base = wild_p(K, [], B=2999, seed=7)
log(f"  기준 (통제없음) β={b_base:+.2f} wild-p={p_base:.4f}")
log(f"  구조6 투입      β={b_obs:+.2f} wild-p={p_obs:.4f}   (β 축소 {(1-abs(b_obs/b_base))*100:.0f}%)")
out["observed"] = {"beta": b_obs, "wild_p": p_obs, "beta_base": b_base}

# ══════════════════════════════════════════════════════════════
log("\n[2] ★ 플라시보 귀무분포 — 자유도 소모는 같고 정보만 다른 통제 6개")
NPLA = 400
FIN = os.path.join(FUS, "ctrl_financial.csv")
pools = {}
pools["(a) 순수 난수"] = None
pools["(b) 구조6 열별 순열"] = "perm"
if os.path.exists(FIN):
    F = pd.read_csv(FIN, encoding="utf-8-sig", dtype={"ksic": str, "k3": str})
    fv = [c for c in F.columns if c.startswith("fin_") and F[c].dtype != object]
    KF = K.merge(F.groupby("k3")[fv].mean().reset_index(), on="k3", how="left")
    fv = [c for c in fv if KF[c].notna().all() and KF[c].std() > 1e-9]
    if len(fv) >= 6:
        pools["(c) 실제 재무지표 6개 무작위추출"] = ("fin", KF, fv)

for lab, mode in pools.items():
    ps, bs = [], []
    for i in range(NPLA):
        d2 = K.copy()
        if mode is None:
            for j in range(len(V)):
                d2[f"z{j}"] = rng.normal(size=len(d2))
            cs = [f"z{j}" for j in range(len(V))]
        elif mode == "perm":
            for j, v in enumerate(V):
                d2[f"z{j}"] = rng.permutation(K[v].values)
            cs = [f"z{j}" for j in range(len(V))]
        else:
            _, KF_, fv_ = mode
            d2 = KF_.copy()
            cs = list(rng.choice(fv_, size=len(V), replace=False))
        try:
            b, p = wild_p(d2, cs, B=399)
            ps.append(p); bs.append(b)
        except Exception:
            pass
    ps, bs = np.array(ps), np.array(bs)
    tail = float((ps >= p_obs).mean())
    btail = float((np.abs(bs) <= abs(b_obs)).mean())
    log(f"\n  {lab}  (플라시보 {len(ps)}회)")
    log(f"    wild-p 귀무분포: 중앙 {np.median(ps):.3f}  90%분위 {np.percentile(ps,90):.3f}  "
        f"5%미달 비율 {(ps<.05).mean():.1%}")
    log(f"    β 귀무분포:     중앙 {np.median(bs):+.2f}  (관측 {b_obs:+.2f})")
    log(f"    ★ 관측 p={p_obs:.3f} 이상인 플라시보 비율 = {tail:.3f}")
    log(f"    ★ 관측 |β| 이하인 플라시보 비율      = {btail:.3f}")
    verdict = ("공유분산 — 구조6은 무작위 통제보다 유의하게 더 많이 죽인다"
               if tail < 0.10 else
               "자유도 고갈 — 무작위 통제도 비슷하게 죽인다")
    log(f"    → 판정: {verdict}")
    out[lab] = {"tail_p": tail, "tail_beta": btail, "median_p": float(np.median(ps)),
                "median_beta": float(np.median(bs))}

# ══════════════════════════════════════════════════════════════
log("\n[3] 공유분산의 크기 직접 측정")
for nm, d in (("n=39", D), ("n=23", K)):
    m = smf.ols(f"padi_e ~ {'+'.join(V)}", data=d).fit()
    log(f"  {nm}: R²(PADI~구조6)={m.rsquared:.3f} adjR²={m.rsquared_adj:+.3f} "
        f"F={m.fvalue:.2f} p={m.f_pvalue:.4f}")

with open(os.path.join(FUS, "df_vs_collinearity.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")

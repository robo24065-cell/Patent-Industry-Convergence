# -*- coding: utf-8 -*-
"""
산업구조 방어 재검정 — 자유도를 존중한 사전지정 사양
────────────────────────────────────────────────────────────────
[배경] defense_structure.py 는 n=23 에서 구조지표 6개 동시투입 시 β_PADI 가 죽었다(wild-p=0.331).
       적대검증자가 더 강하게 지적했다: 63개 부분집합 중 n=23 생존은 4개(6%)로 우연 수준이며,
       저자가 고른 '핵심3' 은 20개 조합 중 wild-p 최소값 = **사후 argmax 보고**였다.
       그 지적은 타당하다. 사양 탐색으로 얻은 생존은 증거가 아니다.

[이 스크립트의 원칙]
  1. 사양을 **사전지정 규칙**으로만 정한다. 부분집합을 뒤져 고르지 않는다.
       - PCA 누적분산 80% 규칙 (재무 방어에서 이미 쓴 것과 동일 규칙)
       - 지표 개수를 고정하지 않고 데이터가 정하게 한다
  2. n=23·G=13 에서 파라미터가 몇 개까지 허용되는지 먼저 밝히고 시작한다.
  3. 결과가 나쁘면 '판정 불가'라고 쓴다. '생존'으로 각색하지 않는다.
"""
import os, json, warnings, itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS  = os.path.join(BASE, "분석", "fusion")
rng  = np.random.default_rng(101)
log  = lambda m: print(m, flush=True)
out  = {}

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str}).dropna(
                subset=["dd_late", "padi_e", "cn_growth_e"])
S = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig", dtype={"ksic": str})
# ctrl_structure.csv 에는 원지표·로그변환·연도수 변형이 섞여 16개가 들어 있다.
# defense_structure.py 가 쓴 6개 정본만 사용한다 (동일 개념의 중복 투입은 다중공선성만 키운다).
SVARS = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
         "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"]
S = S[["ksic"] + [c for c in SVARS if c in S.columns]]
D = P.merge(S, on="ksic", how="left")
SVARS = [c for c in SVARS if c in D.columns]
log(f"결합 {len(D)}산업 | 구조지표 {len(SVARS)}종: {SVARS}")
D = D.dropna(subset=SVARS)
log(f"  결측 제거 후 {len(D)}산업 | 컬럼 확인 k3={'k3' in D.columns} mid={'mid' in D.columns}")


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


def run(df, ctrls, tag):
    fml = "dd_late ~ padi_e + cn_growth_e" + ("".join(f" + {c}" for c in ctrls))
    k = 2 + len(ctrls) + 1
    f = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.mid})
    t, wp = wild_boot(df, fml, "padi_e")
    ok = "유의" if wp < .05 else "비유의"
    log(f"  {tag:30s} n={len(df):3d} k={k:2d} β={f.params.padi_e:+8.2f} "
        f"R²={f.rsquared:.3f} adjR²={f.rsquared_adj:+.3f} wild-p={wp:.4f} {ok}")
    return {"tag": tag, "n": len(df), "k": k, "beta": float(f.params.padi_e),
            "r2": float(f.rsquared), "adj_r2": float(f.rsquared_adj), "wild_p": wp}


# ══════════════════════════════════════════════════════════════
log("\n[0] 자유도 예산 — 시작 전에 밝힌다")
K = D.groupby("k3").agg(dd_late=("dd_late", "first"),
                        **{v: (v, "mean") for v in SVARS + ["padi_e", "cn_growth_e"]},
                        mid=("mid", "first")).reset_index()
for nm, d in (("n=39", D), ("n=23", K)):
    log(f"  {nm}: 관측 {len(d)}, 클러스터 G={d.mid.nunique()}")
log("  클러스터 강건 추론에서 실효 자유도는 G-1 에 가깝다 (n 이 아니라).")
log(f"  → G=13 이면 파라미터 k 가 6~7 을 넘어가면 추론이 신뢰 구간을 잃는다.")
log(f"    구조6 전체투입은 k=9 → 애초에 자유도 예산 밖이었다.")

# ══════════════════════════════════════════════════════════════
log("\n[1] 기준 재현")
res = [run(D, [], "기준 (통제 없음) n=39"), run(K, [], "기준 (통제 없음) n=23")]

log("\n[2] ★ 사전지정 사양 — PCA 누적분산 80% 규칙")
log("  (부분집합 탐색 없음. 재무 방어에서 쓴 것과 동일한 규칙을 그대로 적용.)")
for nm, d in (("n=39", D), ("n=23", K)):
    X = StandardScaler().fit_transform(d[SVARS].values)
    p = PCA().fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    npc = int(np.searchsorted(cum, 0.80) + 1)
    Z = PCA(n_components=npc).fit_transform(X)
    d2 = d.copy()
    for i in range(npc):
        d2[f"spc{i+1}"] = Z[:, i]
    log(f"  [{nm}] 누적분산 80% → 주성분 {npc}개 (설명분산 {cum[npc-1]:.1%})")
    res.append(run(d2, [f"spc{i+1}" for i in range(npc)], f"구조 주성분{npc} {nm}"))

log("\n[3] 사전지정 사양 — 구조지표 1개씩 (자유도 여유 최대)")
for v in SVARS:
    res.append(run(K, [v], f"+{v} (n=23)"))

# ══════════════════════════════════════════════════════════════
log("\n[4] ★ 적대검증자 지적의 정면 대응 — 전수 부분집합 분포")
log("  '핵심3이 통과했다'는 argmax 보고였다는 지적에 대해, 분포 전체를 공개한다.")
for nm, d in (("n=39", D), ("n=23", K)):
    wps, sizes = [], []
    for r_ in range(1, len(SVARS) + 1):
        for combo in itertools.combinations(SVARS, r_):
            fml = "dd_late ~ padi_e + cn_growth_e" + "".join(f" + {c}" for c in combo)
            try:
                _, wp = wild_boot(d, fml, "padi_e", B=999)
                wps.append(wp); sizes.append(r_)
            except Exception:
                pass
    W = pd.DataFrame({"wild_p": wps, "n_ctrl": sizes})
    surv = (W.wild_p < .05).mean()
    log(f"  [{nm}] 부분집합 {len(W)}개 | 5% 생존율 {surv:.1%} | "
        f"wild-p 중앙 {W.wild_p.median():.3f}")
    log(f"        통제 1~2개만: 생존율 {(W[W.n_ctrl<=2].wild_p<.05).mean():.1%} "
        f"| 통제 5~6개: {(W[W.n_ctrl>=5].wild_p<.05).mean():.1%}")
    out[f"subset_{nm}"] = {"n": len(W), "survival": float(surv),
                           "median_p": float(W.wild_p.median()),
                           "surv_low_k": float((W[W.n_ctrl <= 2].wild_p < .05).mean()),
                           "surv_high_k": float((W[W.n_ctrl >= 5].wild_p < .05).mean())}
    W.to_csv(os.path.join(FUS, f"structure_subsets_{nm.replace('=','')}.csv"),
             index=False, encoding="utf-8-sig")

log("\n  → 통제 개수에 따라 생존율이 갈리면, n=23 의 실패는 'PADI 가 가짜'가 아니라")
log("    '자유도 고갈'이라는 해석이 뒷받침된다. 갈리지 않으면 그 해석은 기각된다.")

pd.DataFrame(res).to_csv(os.path.join(FUS, "structure_pca_results.csv"),
                         index=False, encoding="utf-8-sig")
out["specs"] = res
with open(os.path.join(FUS, "structure_pca_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\n저장 → {FUS}")

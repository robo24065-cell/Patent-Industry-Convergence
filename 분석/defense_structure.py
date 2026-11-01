# -*- coding: utf-8 -*-
"""
심사 반론 방어 — 산업구조 통제변수 투입 후에도 β_PADI 가 살아남는가

  [1] predictive_agg.csv + ctrl_structure.csv 병합
  [2] 기준모형 재현 (dd_late ~ padi_e + cn_growth_e, cluster(mid))
  [3] 구조지표 개별 투입 → β_PADI 변화, ΔR²
  [4] 구조지표 전체 포화모형
  [5] Frisch-Waugh-Lovell — PADI 의 구조 직교 성분만으로 예측되는가
  [6] 구조지표만으로 dd_late 예측 → PADI 의 증분정보는 진짜인가
  [7] 모든 검정을 n=39 원자료 / n=23 k3집계 양쪽에서
  [8] wild cluster bootstrap-t (G=13 소표본 클러스터 대응)

※ 기존 분석 파일은 하나도 수정하지 않는다. 새 파일만 생성.
※ 결과를 각색하지 않는다. β_PADI 가 죽으면 죽었다고 쓴다.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
OUT = os.path.join(BASE, "분석", "fusion", "defense_structure_results.csv")
rng = np.random.default_rng(20260812)
log = lambda m: print(m, flush=True)

# 사용할 구조지표 6종 (로그화가 자연스러운 것은 로그판)
USE = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
       "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"]
CORE3 = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size"]  # 반론이 직접 지목한 자본집약·영세성
B_BOOT = 4999  # wild bootstrap 반복수 (G=13 → 2^13=8192 이므로 충분)

ROWS = []  # StructuredOutput 및 CSV 용 결과 축적


def seed_of(*parts):
    """재현 가능한 시드. 파이썬 내장 hash() 는 프로세스마다 달라져서 쓰면 안 된다."""
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % (2 ** 31)


# ══════════════════════════════════════════════════════════════════
# 공통 유틸 — 빠른 OLS + 클러스터 분산 (statsmodels 와 동일한 소표본 보정)
# ══════════════════════════════════════════════════════════════════
def _prep(d, yname, xnames):
    """설계행렬 구성 (절편 포함)"""
    y = d[yname].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[v].to_numpy(float) for v in xnames])
    return y, X


def ols_cluster(y, X, groups):
    """OLS + cluster-robust VCE. statsmodels 의 use_correction=True 와 동일 보정."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    uq = np.unique(groups)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in uq:
        m = groups == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    n, k, G = len(y), X.shape[1], len(uq)
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = XtX_inv @ meat @ XtX_inv * c
    return b, V, u


def wild_boot_cluster(d, yname, xnames, var, gcol, B=B_BOOT, seed=None):
    """
    wild cluster bootstrap-t (Cameron-Gelbach-Miller). 귀무가설 H0: β_var=0 하에서
    제약모형 잔차에 클러스터 단위 Rademacher 가중치를 곱해 재표집.
    관측·부트스트랩 t 모두 cluster-robust SE 기준으로 계산한다.
    (분석 폴더의 effective_n_check.py 에 있는 wild_boot 은 통상 SE 를 썼는데,
     여기서는 클러스터 SE 로 일관되게 맞춘 버전이다.)
    """
    r = np.random.default_rng(seed) if seed is not None else rng
    groups = d[gcol].to_numpy()
    j = 1 + xnames.index(var)  # 절편이 0번
    y, X = _prep(d, yname, xnames)
    b, V, _ = ols_cluster(y, X, groups)
    t_obs = b[j] / np.sqrt(V[j, j])

    # 제약모형 (var 제외)
    x0 = [v for v in xnames if v != var]
    y0, X0 = _prep(d, yname, x0)
    b0 = np.linalg.pinv(X0.T @ X0) @ (X0.T @ y0)
    fit0 = X0 @ b0
    u0 = y0 - fit0

    uq = np.unique(groups)
    idx = {g: (groups == g) for g in uq}
    ts = np.empty(B)
    for i in range(B):
        w = r.choice([-1.0, 1.0], size=len(uq))
        mult = np.ones(len(y))
        for wi, g in zip(w, uq):
            mult[idx[g]] = wi
        yb = fit0 + u0 * mult
        bb, Vb, _ = ols_cluster(yb, X, groups)
        ts[i] = bb[j] / np.sqrt(Vb[j, j])
    p = float((np.abs(ts) >= abs(t_obs) - 1e-12).mean())
    return float(t_obs), p


def fit_sm(d, formula):
    """보고용 statsmodels 적합 (cluster(mid))"""
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.mid})


def rec(spec, n, bp, pp, bc, pc, r2, dr2):
    ROWS.append(dict(spec=spec, n=int(n), beta_padi=float(bp), p_padi=float(pp),
                     beta_ctrl=float(bc), p_ctrl=float(pc),
                     r2=float(r2), delta_r2=float(dr2)))


# ══════════════════════════════════════════════════════════════════
# [1] 병합
# ══════════════════════════════════════════════════════════════════
log("=" * 78)
log("[1] 데이터 병합")
P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
C = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
D = P.merge(C[["ksic"] + USE], on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).reset_index(drop=True)
log(f"  predictive_agg {P.shape} + ctrl_structure {C.shape}")
log(f"  병합 후 n={len(D)}, 구조변수 결측={int(D[USE].isna().sum().sum())}개, "
    f"mid 클러스터={D.mid.nunique()}개, k3={D.k3.nunique()}개")


def make_k3(d, extra):
    """k3(3자리) 집계 — 종속변수는 k3 수준이므로 first, 설명변수는 단순평균"""
    a = d.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                            cn_growth_e=("cn_growth_e", "mean"),
                            mid=("mid", "first")).reset_index()
    ex = d.groupby("k3")[extra].mean().reset_index()
    return a.merge(ex, on="k3")


A = make_k3(D, USE)
SAMPLES = [("n=39", D), ("n=23", A)]
log(f"  k3 집계 표본: n={len(A)}, mid 클러스터={A.mid.nunique()}개")

# ══════════════════════════════════════════════════════════════════
# [2] 기준모형 재현
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[2] 기준모형 재현  dd_late ~ padi_e + cn_growth_e,  cov=cluster(mid)")
BASE_FIT, BASE_R2 = {}, {}
for nm, d in SAMPLES:
    f = fit_sm(d, "dd_late ~ padi_e + cn_growth_e")
    BASE_FIT[nm], BASE_R2[nm] = f, f.rsquared
    log(f"  {nm}: β_PADI={f.params.padi_e:+8.2f} z={f.tvalues.padi_e:+6.2f} "
        f"p={f.pvalues.padi_e:.4f} | β_CN={f.params.cn_growth_e:+7.3f} "
        f"p={f.pvalues.cn_growth_e:.4f} | R²={f.rsquared:.3f} adj={f.rsquared_adj:.3f}")
    rec(f"[기준] dd_late ~ padi_e + cn_growth_e ({nm})", len(d),
        f.params.padi_e, f.pvalues.padi_e,
        f.params.cn_growth_e, f.pvalues.cn_growth_e, f.rsquared, 0.0)
log("  (지시서 기재값: β_padi=-38.70, β_cn=-0.44, R²=0.333, n=39 / n=23 β=-42.82 R²=0.378)")

# ══════════════════════════════════════════════════════════════════
# [3] 구조지표 개별 투입
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[3] 기준식 + 구조지표 1개씩 → β_PADI 이동폭, ΔR²")
for nm, d in SAMPLES:
    base = BASE_FIT[nm]
    log(f"\n  ── {nm} (기준 β_PADI={base.params.padi_e:+.2f}, R²={base.rsquared:.3f})")
    log(f"     {'통제변수':<20s} {'β_PADI':>9s} {'z':>7s} {'p':>8s} {'Δβ%':>7s} "
        f"{'R²':>6s} {'ΔR²':>7s} | {'β_ctrl':>9s} {'p_ctrl':>7s}")
    for v in USE:
        f = fit_sm(d, f"dd_late ~ padi_e + cn_growth_e + {v}")
        chg = (f.params.padi_e - base.params.padi_e) / abs(base.params.padi_e) * 100
        dr2 = f.rsquared - base.rsquared
        mark = "" if f.pvalues.padi_e < 0.05 else "  ← ★비유의★"
        log(f"     +{v:<19s} {f.params.padi_e:+9.2f} {f.tvalues.padi_e:+7.2f} "
            f"{f.pvalues.padi_e:8.4f} {chg:+7.1f} {f.rsquared:6.3f} {dr2:+7.3f} | "
            f"{f.params[v]:+9.3f} {f.pvalues[v]:7.3f}{mark}")
        rec(f"dd_late ~ padi_e + cn_growth_e + {v} ({nm})", len(d),
            f.params.padi_e, f.pvalues.padi_e, f.params[v], f.pvalues[v],
            f.rsquared, dr2)

# ══════════════════════════════════════════════════════════════════
# [4] 포화모형
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[4] 구조지표 동시 투입 (핵심3 / 전체6)")
SATS = [("핵심3", CORE3), ("전체6", USE)]
for nm, d in SAMPLES:
    base = BASE_FIT[nm]
    for tag, S in SATS:
        f = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        chg = (f.params.padi_e - base.params.padi_e) / abs(base.params.padi_e) * 100
        dr2 = f.rsquared - base.rsquared
        dfres = len(d) - (len(S) + 3)
        mark = "" if f.pvalues.padi_e < 0.05 else "  ← ★비유의★"
        log(f"  {nm} {tag:5s}: β_PADI={f.params.padi_e:+8.2f} z={f.tvalues.padi_e:+6.2f} "
            f"p={f.pvalues.padi_e:.4f} Δβ={chg:+6.1f}% R²={f.rsquared:.3f} "
            f"adj={f.rsquared_adj:.3f} ΔR²={dr2:+.3f} (잔차df={dfres}){mark}")
        # 통제변수 중 가장 유의한 것을 대표로 기록
        pc = f.pvalues[S]
        vbest = pc.idxmin()
        rec(f"dd_late ~ padi_e + cn_growth_e + {tag}[{'+'.join(S)}] ({nm}); "
            f"β_ctrl 은 최유의 통제변수 {vbest}", len(d),
            f.params.padi_e, f.pvalues.padi_e, f.params[vbest], f.pvalues[vbest],
            f.rsquared, dr2)

# ══════════════════════════════════════════════════════════════════
# [5] Frisch-Waugh-Lovell
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[5] Frisch-Waugh-Lovell — PADI 의 구조 직교 성분만으로 dd_late 를 예측하는가")
log("    (a) 정통 FWL: y·PADI 를 모두 [구조6 + CN] 에 회귀한 잔차끼리 회귀")
log("        → 계수는 포화모형의 β_PADI 와 수학적으로 동일해야 한다(검산용)")
log("    (b) 부분 잔차화: PADI 만 구조6 에 직교화 후 기준식에 투입")
for nm, d in SAMPLES:
    d = d.copy()
    Z = " + ".join(USE + ["cn_growth_e"])
    d["y_r"] = smf.ols("dd_late ~ " + Z, data=d).fit().resid
    d["p_r"] = smf.ols("padi_e ~ " + Z, data=d).fit().resid
    fa = smf.ols("y_r ~ p_r - 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.mid})
    full = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(USE))
    # 부분 잔차화 (PADI 에서 구조성분만 제거)
    d["p_r2"] = smf.ols("padi_e ~ " + " + ".join(USE), data=d).fit().resid
    fb = fit_sm(d, "dd_late ~ p_r2 + cn_growth_e")
    base = BASE_FIT[nm]
    log(f"\n  ── {nm}")
    log(f"     (a) FWL 잔차회귀 : β={fa.params.p_r:+8.2f} z={fa.tvalues.p_r:+6.2f} "
        f"p={fa.pvalues.p_r:.4f}   [포화모형 β_PADI={full.params.padi_e:+.2f} — "
        f"{'일치' if abs(fa.params.p_r - full.params.padi_e) < 1e-6 else '불일치'}]")
    log(f"         부분 R²(잔차 y 를 잔차 PADI 로 설명) = {fa.rsquared:.3f}")
    log(f"     (b) 부분 잔차화  : β={fb.params.p_r2:+8.2f} z={fb.tvalues.p_r2:+6.2f} "
        f"p={fb.pvalues.p_r2:.4f} R²={fb.rsquared:.3f}  "
        f"(기준 β={base.params.padi_e:+.2f} 대비 크기 "
        f"{fb.params.p_r2 / base.params.padi_e * 100:.1f}%)")
    rec(f"[FWL-a] 잔차(dd_late|구조6+CN) ~ 잔차(padi|구조6+CN) ({nm})", len(d),
        fa.params.p_r, fa.pvalues.p_r, 0.0, 1.0, fa.rsquared,
        fa.rsquared - BASE_R2[nm])
    rec(f"[FWL-b] dd_late ~ resid(padi|구조6) + cn_growth_e ({nm})", len(d),
        fb.params.p_r2, fb.pvalues.p_r2,
        fb.params.cn_growth_e, fb.pvalues.cn_growth_e,
        fb.rsquared, fb.rsquared - BASE_R2[nm])

# ══════════════════════════════════════════════════════════════════
# [6] 구조지표만으로 dd_late 를 얼마나 설명하는가 (PADI 증분정보 검증)
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[6] PADI 없이 구조지표만으로 dd_late 를 설명할 수 있는가")
for nm, d in SAMPLES:
    log(f"\n  ── {nm}")
    specs = [
        ("구조6만            ", "dd_late ~ " + " + ".join(USE)),
        ("구조6 + CN         ", "dd_late ~ cn_growth_e + " + " + ".join(USE)),
        ("구조6 + CN + PADI  ", "dd_late ~ padi_e + cn_growth_e + " + " + ".join(USE)),
        ("핵심3 + CN         ", "dd_late ~ cn_growth_e + " + " + ".join(CORE3)),
        ("핵심3 + CN + PADI  ", "dd_late ~ padi_e + cn_growth_e + " + " + ".join(CORE3)),
        ("PADI + CN (기준)   ", "dd_late ~ padi_e + cn_growth_e"),
        ("PADI 단독          ", "dd_late ~ padi_e"),
    ]
    store = {}
    for tag, fm in specs:
        f = fit_sm(d, fm)
        store[tag.strip()] = f
        bp = f.params.get("padi_e", np.nan)
        pp = f.pvalues.get("padi_e", np.nan)
        bs = "  —      —    " if np.isnan(bp) else f"β_PADI={bp:+7.2f} p={pp:.4f}"
        log(f"     {tag} R²={f.rsquared:6.3f} adj={f.rsquared_adj:+6.3f} "
            f"k={int(f.df_model)}  {bs}")
    s6 = store["구조6 + CN"]; s6p = store["구조6 + CN + PADI"]
    c3 = store["핵심3 + CN"]; c3p = store["핵심3 + CN + PADI"]
    log(f"     → PADI 증분 R² (전체6 위에): {s6p.rsquared - s6.rsquared:+.3f}  "
        f"(구조6+CN 자체 R²={s6.rsquared:.3f}, adj={s6.rsquared_adj:.3f})")
    log(f"     → PADI 증분 R² (핵심3 위에): {c3p.rsquared - c3.rsquared:+.3f}  "
        f"(핵심3+CN 자체 R²={c3.rsquared:.3f}, adj={c3.rsquared_adj:.3f})")
    log(f"     → 역방향 비교: 구조6 증분 R² (기준식 위에) = "
        f"{s6p.rsquared - BASE_R2[nm]:+.3f}")
    rec(f"[구조만] dd_late ~ 구조6 + cn_growth_e (PADI 미포함) ({nm}); β_PADI 는 해당없음(0)",
        len(d), 0.0, 1.0, s6.params.cn_growth_e, s6.pvalues.cn_growth_e,
        s6.rsquared, s6.rsquared - BASE_R2[nm])
    rec(f"[구조만] dd_late ~ 핵심3 + cn_growth_e (PADI 미포함) ({nm}); β_PADI 는 해당없음(0)",
        len(d), 0.0, 1.0, c3.params.cn_growth_e, c3.pvalues.cn_growth_e,
        c3.rsquared, c3.rsquared - BASE_R2[nm])

# ══════════════════════════════════════════════════════════════════
# [8] wild cluster bootstrap-t
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log(f"[8] wild cluster bootstrap-t (Rademacher, B={B_BOOT}, 클러스터=mid)")
log("    G=13 소표본 클러스터에서 정규근사 p 는 과대신뢰. 부트스트랩 p 가 진짜 기준.")
BOOT = {}
boot_specs = [("기준식", ["padi_e", "cn_growth_e"])] + \
             [(f"+{v}", ["padi_e", "cn_growth_e", v]) for v in USE] + \
             [("핵심3", ["padi_e", "cn_growth_e"] + CORE3),
              ("전체6", ["padi_e", "cn_growth_e"] + USE)]
for nm, d in SAMPLES:
    log(f"\n  ── {nm} (G={d.mid.nunique()})")
    for tag, xs in boot_specs:
        t, p = wild_boot_cluster(d, "dd_late", xs, "padi_e", "mid",
                                 B=B_BOOT, seed=seed_of(nm, tag))
        BOOT[(nm, tag)] = (t, p)
        mark = "유의" if p < 0.05 else ("경계(10%)" if p < 0.10 else "★비유의★")
        log(f"     {tag:<20s} t={t:+6.2f}  wild-p={p:.4f}  {mark}")

# ── PADI 잔차(FWL-b) 의 부트스트랩 ────────────────────────────────
log("\n  ── FWL 잔차 사양의 부트스트랩")
for nm, d in SAMPLES:
    d = d.copy()
    d["p_r2"] = smf.ols("padi_e ~ " + " + ".join(USE), data=d).fit().resid
    t, p = wild_boot_cluster(d, "dd_late", ["p_r2", "cn_growth_e"], "p_r2", "mid",
                             B=B_BOOT, seed=seed_of(nm, "fwl"))
    mark = "유의" if p < 0.05 else ("경계(10%)" if p < 0.10 else "★비유의★")
    log(f"     {nm} resid(PADI|구조6): t={t:+6.2f}  wild-p={p:.4f}  {mark}")
    BOOT[(nm, "FWL-b")] = (t, p)

# ══════════════════════════════════════════════════════════════════
# 종합 판정
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[판정] 통상 p / wild-p 동시 기준으로 5% 통과 여부")
verdict = {}
for nm, d in SAMPLES:
    base = BASE_FIT[nm]
    ok, ng = [], []
    for v in USE:
        f = fit_sm(d, f"dd_late ~ padi_e + cn_growth_e + {v}")
        wp = BOOT[(nm, f"+{v}")][1]
        (ok if (f.pvalues.padi_e < 0.05 and wp < 0.05) else ng).append(v)
    verdict[nm] = (ok, ng)
    log(f"  {nm}: 통과 {len(ok)}/6 → {ok}")
    log(f"        탈락 {len(ng)}/6 → {ng}")
    for tag in ("핵심3", "전체6"):
        S = CORE3 if tag == "핵심3" else USE
        f = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        wp = BOOT[(nm, tag)][1]
        log(f"        {tag} 동시투입: p={f.pvalues.padi_e:.4f} wild-p={wp:.4f} "
            f"{'통과' if (f.pvalues.padi_e < 0.05 and wp < 0.05) else '★탈락★'}")

# ══════════════════════════════════════════════════════════════════
# [9] 대칭 검정 — 구조블록은 PADI 위에서 유의한가 (역방향 증분검정)
#     [3]~[6] 은 "구조를 통제하면 PADI 가 사나" 만 봤다. 반대 방향도 봐야
#     어느 쪽이 잉여인지 판정할 수 있다. 클러스터-로버스트 결합 Wald.
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[9] 역방향 결합검정 — PADI+CN 이 이미 있을 때 구조블록이 추가 설명을 하는가")
log("    (cluster(mid) 로버스트 Wald, H0: 해당 블록 계수 전부 0)")
for nm, d in SAMPLES:
    log(f"\n  ── {nm}")
    for tag, S in SATS:
        f = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        w = f.f_test(" = 0, ".join(S) + " = 0")  # 구조블록 결합영가설
        fp = float(np.squeeze(w.pvalue))
        # 대칭 비교용: 같은 모형에서 PADI 단독 계수의 p
        log(f"     {tag:5s}: 구조블록 결합 p={fp:.4f} {'유의' if fp < 0.05 else '★비유의★'}"
            f"   |  같은 모형 β_PADI p={f.pvalues.padi_e:.4f}")
        rec(f"[역방향] {tag} 구조블록 결합 Wald (PADI+CN 통제 후) ({nm}); "
            f"beta_ctrl/p_ctrl 은 블록결합검정 통계량/p", len(d),
            f.params.padi_e, f.pvalues.padi_e,
            float(np.squeeze(w.fvalue)), fp,
            f.rsquared, f.rsquared - BASE_R2[nm])

# ══════════════════════════════════════════════════════════════════
# [10] 클러스터 잭나이프 — 권고사양(핵심3)이 특정 중분류 하나에 기대고 있는가
#      n=23 핵심3 은 wild-p 가 5% 경계에 있어 취약성 점검이 필수다.
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 78)
log("[10] leave-one-mid-out 잭나이프 — 권고사양 dd_late ~ PADI + CN + 핵심3")
log("     중분류 13개를 하나씩 빼고 재추정. 한 클러스터에 의존하면 여기서 드러난다.")
for nm, d in SAMPLES:
    bs, ps = [], []
    worst = None
    for g in sorted(d.mid.unique()):
        sub = d[d.mid != g]
        if sub.mid.nunique() < 3:
            continue
        f = fit_sm(sub, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(CORE3))
        bs.append(f.params.padi_e); ps.append(f.pvalues.padi_e)
        if worst is None or f.pvalues.padi_e > worst[1]:
            worst = (g, f.pvalues.padi_e, f.params.padi_e)
    bs, ps = np.array(bs), np.array(ps)
    nfail = int((ps >= 0.05).sum())
    log(f"  {nm}: β_PADI 범위 [{bs.min():+.2f}, {bs.max():+.2f}] (중앙 {np.median(bs):+.2f}), "
        f"p 최대 {ps.max():.4f}")
    log(f"        5% 미달 폴드 {nfail}/{len(ps)}개"
        + (f"  ← 최악 폴드: mid={worst[0]} 제외 시 β={worst[2]:+.2f} p={worst[1]:.4f}"
           if nfail else "  → 어떤 중분류를 빼도 유의성 유지"))

R = pd.DataFrame(ROWS)
R.to_csv(OUT, index=False, encoding="utf-8-sig")
log(f"\n[저장] {OUT}  ({len(R)}행)")

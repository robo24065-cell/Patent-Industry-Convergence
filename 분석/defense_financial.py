# -*- coding: utf-8 -*-
"""
심사 반론 방어 (재무편) — "PADI 는 그냥 돈 없는 산업을 잡아낸 것 아닌가?"
                          재무 통제변수 투입 후에도 β_PADI 가 살아남는가.

  [1] predictive_agg.csv + ctrl_financial.csv 병합, 매칭률
  [2] 기준모형 재현 (dd_late ~ padi_e + cn_growth_e, cluster(mid))
  [3] 재무지표 1개씩 투입 → β_PADI 계수·유의도 변화, ΔR²
  [4] 포화모형 (핵심4 / 전체9 / 전체10)
  [5] VIF 다중공선성 + 해상도 진단(k3 vs mid within 변동)
  [6] ★ 결정적 검정: Frisch-Waugh-Lovell — 재무로 설명되지 않는 PADI 성분만으로 예측되는가
  [7] 모든 검정을 n=39 원자료 / n=23 k3집계 양쪽에서
  [8] wild cluster bootstrap-t (Rademacher, mid 클러스터, B>=2999)
  [9] 대안사양 — 클러스터를 k3 로 바꾼 버전 / 재무지표를 mid해상도로 넣은 대조

※ 기존 분석 파일은 하나도 수정하지 않는다. 새 파일만 생성.
※ 결과를 각색하지 않는다. β_PADI 가 죽으면 죽었다고 쓴다.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
OUT = os.path.join(FUS, "defense_financial_results.csv")
rng = np.random.default_rng(20260812)
log = lambda m: print(m, flush=True)

# ── 재무지표 세트 정의 ────────────────────────────────────────────
# 9종 기본지표 (ctrl_financial.csv 의 fin_*_mid 대응쌍이 존재하는 것들)
FIN9 = ["fin_debt_ratio", "fin_equity_ratio", "fin_current_ratio", "fin_borrow_dep",
        "fin_opm", "fin_icr", "fin_int_rate", "fin_rnd_sales", "fin_sales_growth"]
# icr_low 는 fin_icr 의 유계변환(한계기업 비중). k3 23개 중 5개만 비영 → 보조지표
FIN10 = FIN9 + ["fin_icr_low"]
# "돈 없는 산업" 반론이 직접 지목하는 것: 레버리지·조달비용·수익성·성장
CORE4 = ["fin_debt_ratio", "fin_borrow_dep", "fin_opm", "fin_sales_growth"]
# VIF 정제판: 자기자본비율(부채비율과 회계항등식상 중복) + 이자보상비율 수준(우측꼬리 극심) 제거
FIN7 = ["fin_debt_ratio", "fin_current_ratio", "fin_borrow_dep", "fin_opm",
        "fin_int_rate", "fin_rnd_sales", "fin_sales_growth"]
NPC = 3  # 재무 주성분 개수 (n=23 자유도 절약 사양)

B_BOOT = 4999   # G=13 → 2^13=8192 이므로 4999 는 충분
ROWS = []       # 결과 축적 (CSV + StructuredOutput)


def seed_of(*parts):
    """재현 가능한 시드. 파이썬 내장 hash() 는 프로세스마다 달라져 쓰면 안 된다."""
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % (2 ** 31)


# ══════════════════════════════════════════════════════════════════
# 공통 유틸 — OLS + 클러스터 분산 (statsmodels use_correction=True 와 동일 보정)
# ══════════════════════════════════════════════════════════════════
def _prep(d, yname, xnames):
    y = d[yname].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[v].to_numpy(float) for v in xnames])
    return y, X


def ols_cluster(y, X, groups):
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
    wild cluster bootstrap-t (Cameron-Gelbach-Miller 2008).
    귀무 H0: β_var=0 하의 제약모형 잔차에 클러스터 단위 Rademacher 가중치를 곱해 재표집.
    관측 t 와 부트스트랩 t 를 모두 cluster-robust SE 로 계산해 일관성을 맞춘다.
    (effective_n_check.py 의 wild_boot 은 통상 SE 를 썼다 — 여기서는 클러스터 SE 판.)
    """
    r = np.random.default_rng(seed) if seed is not None else rng
    groups = d[gcol].to_numpy()
    j = 1 + xnames.index(var)          # 절편이 0번
    y, X = _prep(d, yname, xnames)
    b, V, _ = ols_cluster(y, X, groups)
    t_obs = b[j] / np.sqrt(V[j, j])

    x0 = [v for v in xnames if v != var]        # 제약모형 (var 제외)
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


def fit_sm(d, formula, gcol="mid"):
    """보고용 statsmodels 적합 (기본 cluster(mid))"""
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[gcol]})


def rec(spec, n, bp, pp, bc, pc, r2, dr2):
    ROWS.append(dict(spec=spec, n=int(n), beta_padi=float(bp), p_padi=float(pp),
                     beta_ctrl=float(bc), p_ctrl=float(pc),
                     r2=float(r2), delta_r2=float(dr2)))


def mark_p(p):
    return "" if p < 0.05 else ("  ← 경계(10%)" if p < 0.10 else "  ← ★비유의★")


# ══════════════════════════════════════════════════════════════════
# [1] 병합 · 매칭률
# ══════════════════════════════════════════════════════════════════
log("=" * 82)
log("[1] 데이터 병합 및 매칭률")
P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
C = pd.read_csv(os.path.join(FUS, "ctrl_financial.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
log(f"  predictive_agg.csv {P.shape} / ctrl_financial.csv {C.shape}")

MIDV = [v + "_mid" for v in FIN9]
keep = ["ksic"] + FIN10 + MIDV + ["fin_res", "fin_src_code"]
D = P.merge(C[keep], on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).reset_index(drop=True)

matched = int(D[FIN10].notna().all(axis=1).sum())
log(f"  병합 후 n={len(D)}  |  재무 10종 전부 비결측 행 = {matched}/{len(D)} "
    f"({matched / len(D) * 100:.1f}%)")
for v in FIN10:
    nn = int(D[v].notna().sum())
    if nn < len(D):
        log(f"    ! 결측 있음: {v} = {nn}/{len(D)}")
log(f"  결측 총계 = {int(D[FIN10].isna().sum().sum())}개")
log(f"  mid 클러스터 = {D.mid.nunique()}개, k3 = {D.k3.nunique()}개, "
    f"고유 ECOS 계열 = {D.fin_src_code.nunique()}개")
log("  해상도 내역 (fin_res: 3=3자리단독, 31=3자리병합군, 2=2자리폴백)")
for r_, g in D.groupby("fin_res"):
    log(f"    res={int(r_):>2d}: 행 {len(g):2d}개, k3 {g.k3.nunique():2d}개 "
        f"→ {sorted(g.k3.unique())}")

log("\n  재무지표 기술통계 (n=39)")
log(f"    {'변수':<20s} {'평균':>9s} {'SD':>8s} {'최소':>9s} {'최대':>9s}")
for v in FIN10:
    s = D[v]
    log(f"    {v:<20s} {s.mean():9.3f} {s.std():8.3f} {s.min():9.3f} {s.max():9.3f}")


def make_k3(d, extra):
    """k3(3자리) 집계 — 종속변수는 k3 수준이므로 first, 설명변수는 단순평균"""
    a = d.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                            cn_growth_e=("cn_growth_e", "mean"),
                            mid=("mid", "first")).reset_index()
    ex = d.groupby("k3")[extra].mean().reset_index()
    return a.merge(ex, on="k3")


A = make_k3(D, FIN10 + MIDV)
log(f"\n  k3 집계 표본: n={len(A)}, mid 클러스터={A.mid.nunique()}개")


# ── 재무 주성분 (표준화 후 SVD). n=23 에 통제 9개를 태우면 자유도가 고갈되므로
#    재무정보를 소수 축으로 압축한 절약사양을 함께 본다.
#    ※ 적재(loading)는 n=39 표본에서 한 번만 추정하고 k3 집계표본에 그대로 투영한다.
def add_pcs(d, mu, sd, V, npc=NPC):
    Z = (d[FIN9].to_numpy(float) - mu) / sd
    S = Z @ V[:npc].T
    for i in range(npc):
        d[f"fin_pc{i + 1}"] = S[:, i]
    return d


_M = D[FIN9].to_numpy(float)
_mu, _sd = _M.mean(0), _M.std(0, ddof=1)
_U, _s, _Vt = np.linalg.svd((_M - _mu) / _sd, full_matrices=False)
_evr = _s ** 2 / (_s ** 2).sum()
D = add_pcs(D, _mu, _sd, _Vt)
A = add_pcs(A, _mu, _sd, _Vt)
PCS = [f"fin_pc{i + 1}" for i in range(NPC)]
log(f"  재무 주성분: 설명분산비 " +
    ", ".join(f"PC{i + 1}={_evr[i]:.3f}" for i in range(NPC)) +
    f" (누적 {_evr[:NPC].sum():.3f})")

SAMPLES = [("n=39", D), ("n=23", A)]

# ══════════════════════════════════════════════════════════════════
# [2] 기준모형 재현
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
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
log("  (지시서 기재값: n39 β_padi=-38.70 β_cn=-0.44 R²=0.333 / n23 β_padi=-42.82 R²=0.378)")

# ── PADI 와 재무지표의 단순상관 (반론의 이변량 근거 점검) ──────────
log("\n  PADI vs 재무지표 단순상관 (반론이 맞다면 강한 상관이 있어야 한다)")
log(f"    {'변수':<20s} {'r(n=39)':>9s} {'r(n=23)':>9s} {'r(재무,dd_late) n39':>20s}")
for v in FIN10:
    r39 = D.padi_e.corr(D[v]); r23 = A.padi_e.corr(A[v])
    ry = D[v].corr(D.dd_late)
    log(f"    {v:<20s} {r39:+9.3f} {r23:+9.3f} {ry:+20.3f}")

# ══════════════════════════════════════════════════════════════════
# [3] 재무지표 1개씩 투입
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log("[3] 기준식 + 재무지표 1개씩 → β_PADI 이동폭, ΔR²")
for nm, d in SAMPLES:
    base = BASE_FIT[nm]
    log(f"\n  ── {nm} (기준 β_PADI={base.params.padi_e:+.2f}, R²={base.rsquared:.3f})")
    log(f"     {'통제변수':<21s} {'β_PADI':>9s} {'z':>7s} {'p':>8s} {'Δβ%':>7s} "
        f"{'R²':>6s} {'ΔR²':>7s} | {'β_ctrl':>10s} {'p_ctrl':>7s}")
    for v in FIN10:
        f = fit_sm(d, f"dd_late ~ padi_e + cn_growth_e + {v}")
        chg = (f.params.padi_e - base.params.padi_e) / abs(base.params.padi_e) * 100
        dr2 = f.rsquared - base.rsquared
        log(f"     +{v:<20s} {f.params.padi_e:+9.2f} {f.tvalues.padi_e:+7.2f} "
            f"{f.pvalues.padi_e:8.4f} {chg:+7.1f} {f.rsquared:6.3f} {dr2:+7.3f} | "
            f"{f.params[v]:+10.4f} {f.pvalues[v]:7.3f}{mark_p(f.pvalues.padi_e)}")
        rec(f"dd_late ~ padi_e + cn_growth_e + {v} ({nm})", len(d),
            f.params.padi_e, f.pvalues.padi_e, f.params[v], f.pvalues[v],
            f.rsquared, dr2)

# ══════════════════════════════════════════════════════════════════
# [4] 포화모형
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log("[4] 재무지표 동시 투입 (포화모형)")
SATS = [("핵심4", CORE4), ("정제7", FIN7), ("전체9", FIN9), ("전체10", FIN10),
        ("주성분3", PCS)]
for nm, d in SAMPLES:
    base = BASE_FIT[nm]
    log(f"\n  ── {nm}")
    for tag, S in SATS:
        f = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        chg = (f.params.padi_e - base.params.padi_e) / abs(base.params.padi_e) * 100
        dr2 = f.rsquared - base.rsquared
        dfres = len(d) - (len(S) + 3)
        log(f"     {tag:6s}: β_PADI={f.params.padi_e:+8.2f} z={f.tvalues.padi_e:+6.2f} "
            f"p={f.pvalues.padi_e:.4f} Δβ={chg:+6.1f}% R²={f.rsquared:.3f} "
            f"adj={f.rsquared_adj:+.3f} ΔR²={dr2:+.3f} (잔차df={dfres})"
            f"{mark_p(f.pvalues.padi_e)}")
        pc = f.pvalues[S]
        vbest = pc.idxmin()
        rec(f"dd_late ~ padi_e + cn_growth_e + {tag}[{'+'.join(S)}] ({nm}); "
            f"β_ctrl 은 최유의 통제변수 {vbest}", len(d),
            f.params.padi_e, f.pvalues.padi_e, f.params[vbest], f.pvalues[vbest],
            f.rsquared, dr2)
        sig = [f"{k}(p={pc[k]:.3f})" for k in S if pc[k] < 0.10]
        log(f"             통제변수 중 p<0.10: {sig if sig else '없음'}")

    # PADI 를 뺀 재무만 모형 — 재무의 자체 설명력
    for tag, S in SATS:
        f0 = fit_sm(d, "dd_late ~ cn_growth_e + " + " + ".join(S))
        fp = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        log(f"     [{tag}] 재무+CN 만 R²={f0.rsquared:.3f} (adj={f0.rsquared_adj:+.3f}) "
            f"→ PADI 증분 R² = {fp.rsquared - f0.rsquared:+.3f}")
        rec(f"[재무만] dd_late ~ {tag}+cn_growth_e (PADI 미포함) ({nm}); β_PADI 해당없음(0)",
            len(d), 0.0, 1.0, f0.params.cn_growth_e, f0.pvalues.cn_growth_e,
            f0.rsquared, f0.rsquared - BASE_R2[nm])

# ══════════════════════════════════════════════════════════════════
# [5] VIF + 해상도 진단
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log("[5] 다중공선성 VIF (설계행렬 = 절편 + padi + cn + 재무세트)")
for nm, d in SAMPLES:
    for tag, S in SATS:
        cols = ["padi_e", "cn_growth_e"] + S
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
        vifs = [variance_inflation_factor(X, i + 1) for i in range(len(cols))]
        worst = max(zip(vifs, cols))
        log(f"\n  ── {nm} / {tag} (k={len(cols)}, n={len(d)})  최대 VIF={worst[0]:.1f} ({worst[1]})")
        log("     " + "  ".join(f"{c}={v:.1f}" for c, v in zip(cols, vifs)))
        bad = [c for c, v in zip(cols, vifs) if v > 10]
        if bad:
            log(f"     ★ VIF>10 : {bad}  ← 계수 개별해석 불가 구간")

log("\n  ※ VIF 상 부채비율/자기자본비율은 회계항등식상 거의 역수관계 → 동시투입 시 폭증 예상")
log("\n  해상도 진단 — within-mid 표준편차 (0 이면 클러스터 내 변동이 없어 검정력 소멸)")
log(f"    {'변수':<20s} {'k3판 within-SD':>16s} {'mid판 within-SD':>17s}")
for v in FIN9:
    w3 = D.groupby("mid")[v].std(ddof=0).mean()
    wm = D.groupby("mid")[v + "_mid"].std(ddof=0).mean()
    log(f"    {v:<20s} {w3:16.4f} {wm:17.4f}")
log("    → k3 해상도 확보 덕분에 within 변동이 남아있다 (수입침투율은 여기서 0 이라 탈락했다)")

# ══════════════════════════════════════════════════════════════════
# [6] ★ 결정적 검정: Frisch-Waugh-Lovell
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log("[6] ★ FWL — 재무로 설명되지 않는 PADI 성분만으로 dd_late 를 예측할 수 있는가")
log("    (a) 정통 FWL: dd_late·PADI 를 모두 [재무 + CN] 에 회귀한 잔차끼리 회귀")
log("        → 계수는 포화모형 β_PADI 와 수학적으로 동일해야 한다(검산)")
log("""    (b) 부분 잔차화: PADI 만 재무세트에 직교화 후 기준식에 투입
        ※ (b) 는 설계상 (a) 보다 반드시 보수적이다. PADI 에서만 재무성분을 빼고
          dd_late 에서는 빼지 않으므로 dd_late 의 재무성분이 오차항에 그대로 남아
          SE 가 부풀려진다. 즉 (b) 의 큰 p 값은 '효과가 없다'가 아니라
          '이 사양이 검정력을 버렸다'는 뜻이다. 두 값을 모두 보고한다.""")
FWL_STORE = {}
for nm, d0 in SAMPLES:
    log(f"\n  ══ {nm}")
    for tag, S in SATS:
        d = d0.copy()
        Z = " + ".join(S + ["cn_growth_e"])
        d["y_r"] = smf.ols("dd_late ~ " + Z, data=d).fit().resid
        d["p_r"] = smf.ols("padi_e ~ " + Z, data=d).fit().resid
        fa = smf.ols("y_r ~ p_r - 1", data=d).fit(cov_type="cluster",
                                                  cov_kwds={"groups": d.mid})
        full = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        ok = "일치" if abs(fa.params.p_r - full.params.padi_e) < 1e-6 else "불일치"

        # 재무가 PADI 분산을 얼마나 먹는가 (★ adj R² 와 F검정을 함께 봐야 한다.
        #   통제 9개 대 n=23 에서는 R² 가 자동으로 부풀기 때문에 원 R² 만 보면 오독한다)
        fp = smf.ols("padi_e ~ " + " + ".join(S), data=d).fit()
        d["p_r2"] = fp.resid
        fb = fit_sm(d, "dd_late ~ p_r2 + cn_growth_e")
        base = BASE_FIT[nm]
        FWL_STORE[(nm, tag)] = d[["p_r2"]].copy()

        log(f"    ── {tag}  [재무가 설명하는 PADI 분산: R²={fp.rsquared:.3f} "
            f"adj={fp.rsquared_adj:+.3f} F={fp.fvalue:.2f} p={fp.f_pvalue:.3f}"
            f"{'  ← adj<0: 재무는 PADI 를 잡음 이상으로 설명하지 못한다' if fp.rsquared_adj < 0 else ''}]")
        log(f"       (a) FWL 잔차회귀: β={fa.params.p_r:+8.2f} z={fa.tvalues.p_r:+6.2f} "
            f"p={fa.pvalues.p_r:.4f}  [포화 β={full.params.padi_e:+.2f} — {ok}]")
        log(f"       (b) 부분잔차화  : β={fb.params.p_r2:+8.2f} z={fb.tvalues.p_r2:+6.2f} "
            f"p={fb.pvalues.p_r2:.4f} R²={fb.rsquared:.3f} "
            f"(기준 β={base.params.padi_e:+.2f} 대비 {fb.params.p_r2 / base.params.padi_e * 100:.1f}%)"
            f"{mark_p(fb.pvalues.p_r2)}")
        rec(f"[FWL-b] dd_late ~ resid(padi|{tag}) + cn_growth_e ({nm}); "
            f"R²(PADI~재무)={fp.rsquared:.3f}", len(d),
            fb.params.p_r2, fb.pvalues.p_r2,
            fb.params.cn_growth_e, fb.pvalues.cn_growth_e,
            fb.rsquared, fb.rsquared - BASE_R2[nm])

# ══════════════════════════════════════════════════════════════════
# [8] wild cluster bootstrap-t
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log(f"[8] wild cluster bootstrap-t (Rademacher, B={B_BOOT}, 클러스터=mid)")
log("    G=13 소표본 클러스터에서 정규근사 p 는 과대신뢰. 부트스트랩 p 가 진짜 기준.")
BOOT = {}
boot_specs = [("기준식", ["padi_e", "cn_growth_e"])] + \
             [(f"+{v}", ["padi_e", "cn_growth_e", v]) for v in FIN10] + \
             [(tag, ["padi_e", "cn_growth_e"] + S) for tag, S in SATS]
for nm, d in SAMPLES:
    log(f"\n  ── {nm} (G={d.mid.nunique()})")
    for tag, xs in boot_specs:
        t, p = wild_boot_cluster(d, "dd_late", xs, "padi_e", "mid",
                                 B=B_BOOT, seed=seed_of("fin", nm, tag))
        BOOT[(nm, tag)] = (t, p)
        mk = "유의" if p < 0.05 else ("경계(10%)" if p < 0.10 else "★비유의★")
        log(f"     {tag:<22s} t={t:+6.2f}  wild-p={p:.4f}  {mk}")

log("\n  ── FWL 잔차 사양(b)의 부트스트랩 ★결정적")
for nm, d0 in SAMPLES:
    for tag, S in SATS:
        d = d0.copy()
        d["p_r2"] = smf.ols("padi_e ~ " + " + ".join(S), data=d).fit().resid
        t, p = wild_boot_cluster(d, "dd_late", ["p_r2", "cn_growth_e"], "p_r2", "mid",
                                 B=B_BOOT, seed=seed_of("fin", nm, "fwl", tag))
        BOOT[(nm, "FWL-" + tag)] = (t, p)
        mk = "유의" if p < 0.05 else ("경계(10%)" if p < 0.10 else "★비유의★")
        log(f"     {nm} resid(PADI|{tag:6s}): t={t:+6.2f}  wild-p={p:.4f}  {mk}")

# ══════════════════════════════════════════════════════════════════
# [9] 대안사양 — 클러스터 단위 변경 / 재무 mid해상도 대조
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log("[9] 대안사양 (해상도·클러스터 민감도)")

log("\n  (9-1) 클러스터를 k3 로 바꾼 버전 (G=23, mid 보다 덜 보수적)")
for nm, d in SAMPLES:
    for tag, S in [("기준식", [])] + SATS:
        fm = "dd_late ~ padi_e + cn_growth_e" + ("" if not S else " + " + " + ".join(S))
        f = fit_sm(d, fm, gcol="k3")
        log(f"     {nm} {tag:6s}: β_PADI={f.params.padi_e:+8.2f} "
            f"p={f.pvalues.padi_e:.4f} R²={f.rsquared:.3f}{mark_p(f.pvalues.padi_e)}")

log("\n  (9-2) 재무지표를 mid(2자리) 해상도로 넣은 대조 — 해상도 함정 실증")
log(f"     {'변수':<20s} {'k3판 p_ctrl':>12s} {'mid판 p_ctrl':>13s} "
    f"{'k3판 β_PADI':>12s} {'mid판 β_PADI':>13s}")
for v in FIN9:
    f3 = fit_sm(D, f"dd_late ~ padi_e + cn_growth_e + {v}")
    fmd = fit_sm(D, f"dd_late ~ padi_e + cn_growth_e + {v}_mid")
    log(f"     {v:<20s} {f3.pvalues[v]:12.4f} {fmd.pvalues[v + '_mid']:13.4f} "
        f"{f3.params.padi_e:+12.2f} {fmd.params.padi_e:+13.2f}")

log("\n  (9-3) leave-one-mid-out — 특정 중분류 하나가 결과를 떠받치는가 (핵심4 사양)")
for nm, d in SAMPLES:
    keep_sig, betas = 0, []
    for g in sorted(d.mid.unique()):
        sub = d[d.mid != g]
        if sub.mid.nunique() < 3:
            continue
        f = fit_sm(sub, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(CORE4))
        betas.append(f.params.padi_e)
        keep_sig += int(f.pvalues.padi_e < 0.05)
    log(f"     {nm}: p<0.05 유지 {keep_sig}/{len(betas)} "
        f"({keep_sig / len(betas):.2f}), β_PADI 범위 [{min(betas):+.2f}, {max(betas):+.2f}], "
        f"부호일관 {'예' if all(b < 0 for b in betas) else '아니오'}")

# ══════════════════════════════════════════════════════════════════
# 종합 판정
# ══════════════════════════════════════════════════════════════════
log("\n" + "=" * 82)
log("[판정] 통상 p / wild-p 동시 5% 기준")
for nm, d in SAMPLES:
    ok, ng = [], []
    for v in FIN10:
        f = fit_sm(d, f"dd_late ~ padi_e + cn_growth_e + {v}")
        wp = BOOT[(nm, f"+{v}")][1]
        (ok if (f.pvalues.padi_e < 0.05 and wp < 0.05) else ng).append(v)
    log(f"\n  {nm} 개별투입: 통과 {len(ok)}/10 → {ok}")
    log(f"              탈락 {len(ng)}/10 → {ng}")
    for tag, S in SATS:
        f = fit_sm(d, "dd_late ~ padi_e + cn_growth_e + " + " + ".join(S))
        wp = BOOT[(nm, tag)][1]
        fw = BOOT[(nm, "FWL-" + tag)][1]
        log(f"    {tag:6s} 동시투입: p={f.pvalues.padi_e:.4f} wild-p={wp:.4f} "
            f"{'통과' if (f.pvalues.padi_e < 0.05 and wp < 0.05) else '★탈락★'}"
            f"  | FWL잔차 wild-p={fw:.4f} "
            f"{'통과' if fw < 0.05 else '★탈락★'}")

R = pd.DataFrame(ROWS)
R.to_csv(OUT, index=False, encoding="utf-8-sig")
log(f"\n[저장] {OUT}  ({len(R)}행)")

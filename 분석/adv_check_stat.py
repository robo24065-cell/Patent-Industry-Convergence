# -*- coding: utf-8 -*-
"""
적대적 검증 — 통계적 타당성 렌즈
  defense_structure.py 의 결론("n=39 전부 통과 / n=23 완전 사망")을 반증 시도.
  기존 파일 수정 없음. 신규 파일.
"""
import os, sys, warnings, itertools
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
USE = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
       "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"]
CORE3 = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size"]

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
C = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
D = P.merge(C[["ksic"] + USE], on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).reset_index(drop=True)

def make_k3(d, extra):
    a = d.groupby("k3").agg(dd_late=("dd_late", "first"), padi_e=("padi_e", "mean"),
                            cn_growth_e=("cn_growth_e", "mean"),
                            mid=("mid", "first")).reset_index()
    ex = d.groupby("k3")[extra].mean().reset_index()
    return a.merge(ex, on="k3")

A = make_k3(D, USE)
print("="*80)
print(f"[0] n=39 표본 {D.shape}, 구조변수 결측 {int(D[USE].isna().sum().sum())}")
print(f"    n=23 표본 {A.shape}")

# ── [A] 유사반복(pseudo-replication): y 가 k3 내에서 상수인가 ──────────
print("\n" + "="*80)
print("[A] 종속변수 dd_late 의 k3 내 변동 — n=39 는 진짜 39개 관측인가?")
g = D.groupby("k3")["dd_late"]
within_sd = g.std(ddof=0)
sizes = g.size()
print(f"    k3 그룹 수={len(sizes)}, 그룹크기 분포={sizes.value_counts().to_dict()}")
print(f"    k3 내 dd_late 표준편차 최댓값 = {within_sd.max():.10f}  "
      f"(0 이면 y 가 k3 수준에서 복제된 값)")
print(f"    → 고유한 dd_late 값 개수 = {D.dd_late.round(9).nunique()} / 39")
# padi 는?
print(f"    참고: k3 내 padi_e 표준편차 평균 = {D.groupby('k3')['padi_e'].std(ddof=0).mean():.4f}")
# mid 클러스터 크기
print(f"    mid 클러스터 크기(n=39) = {D.groupby('mid').size().to_dict()}")
print(f"    mid 클러스터 크기(n=23) = {A.groupby('mid').size().to_dict()}")

# ── [B] statsmodels cluster 의 자유도 처리 ────────────────────────────
print("\n" + "="*80)
print("[B] statsmodels cov_type='cluster' 가 t 를 쓰는가 z 를 쓰는가 (자유도 문제)")
for nm, d in [("n=39", D), ("n=23", A)]:
    f = smf.ols("dd_late ~ padi_e + cn_growth_e", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.mid})
    print(f"    {nm}: use_t={f.use_t}, df_resid_used={f.df_resid}, "
          f"t_padi={f.tvalues.padi_e:.3f}, p={f.pvalues.padi_e:.6f}")
    from scipy import stats
    G = d.mid.nunique()
    t = f.tvalues.padi_e
    print(f"          → 정규근사 p={2*(1-stats.norm.cdf(abs(t))):.6f} | "
          f"t(G-1={G-1}) p={2*(1-stats.t.cdf(abs(t), G-1)):.6f}")

# ── [C] 포화모형에서 모수 개수 vs 클러스터 수 ─────────────────────────
print("\n" + "="*80)
print("[C] 포화모형: 모수 k, 클러스터 G, 잔차 df, CRVE 랭크")
from scipy import stats
def crve_diag(d, xs, gcol="mid"):
    y = d["dd_late"].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[v].to_numpy(float) for v in xs])
    groups = d[gcol].to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y); u = y - X @ b
    uq = np.unique(groups)
    S = np.array([X[groups==g].T @ u[groups==g] for g in uq])
    return len(d), X.shape[1], len(uq), np.linalg.matrix_rank(S.T @ S), np.linalg.cond(X)

for nm, d in [("n=39", D), ("n=23", A)]:
    for tag, S in [("기준", ["padi_e","cn_growth_e"]),
                   ("핵심3", ["padi_e","cn_growth_e"]+CORE3),
                   ("전체6", ["padi_e","cn_growth_e"]+USE)]:
        n,k,G,rk,cond = crve_diag(d, S)
        print(f"    {nm} {tag:5s}: n={n} k={k} G={G} 잔차df={n-k} "
              f"CRVE_meat랭크={rk} cond(X)={cond:.1f}  "
              f"{'← k>G-1 위험' if k > G-1 else ''}")

# ── [D] 다중공선성 VIF ────────────────────────────────────────────────
print("\n" + "="*80)
print("[D] 다중공선성 VIF (전체6 포화모형 기준)")
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
for nm, d in [("n=39", D), ("n=23", A)]:
    cols = ["padi_e","cn_growth_e"]+USE
    X = sm.add_constant(d[cols].astype(float).to_numpy())
    vs = {c: vif(X, i+1) for i, c in enumerate(cols)}
    print(f"    {nm}: " + ", ".join(f"{k}={v:.1f}" for k,v in vs.items()))

# ── [E] 구조변수 해상도: within-mid / within-k3 변동 비율 ─────────────
print("\n" + "="*80)
print("[E] 구조지표 해상도 — 총분산 대비 within 분산 비율")
def within_share(d, v, by):
    x = d[v].astype(float)
    tot = ((x - x.mean())**2).sum()
    wi = (x - d.groupby(by)[v].transform("mean")).pow(2).sum()
    return wi/tot if tot>0 else np.nan
for v in USE + ["padi_e", "dd_late"]:
    print(f"    {v:<20s} within-mid(n=39)={within_share(D,v,'mid'):.3f} "
          f"within-k3(n=39)={within_share(D,v,'k3'):.3f} "
          f"within-mid(n=23)={within_share(A,v,'mid'):.3f}")

# ── [F] CR3(클러스터 잭나이프) SE 로 재검정 ───────────────────────────
print("\n" + "="*80)
print("[F] CR3 (leave-one-cluster-out 잭나이프) SE — CR1 이 과소추정인지 확인")
def cr3(d, xs, var, gcol="mid"):
    y = d["dd_late"].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[v].to_numpy(float) for v in xs])
    groups = d[gcol].to_numpy(); uq = np.unique(groups)
    b = np.linalg.pinv(X.T@X)@(X.T@y)
    j = 1 + xs.index(var)
    bs = []
    for g in uq:
        m = groups != g
        try:
            bg = np.linalg.pinv(X[m].T@X[m])@(X[m].T@y[m])
        except Exception:
            continue
        bs.append(bg[j])
    bs = np.array(bs); G = len(bs)
    se = np.sqrt((G-1)/G*((bs-bs.mean())**2).sum())
    return b[j], se, b[j]/se, G

for nm, d in [("n=39", D), ("n=23", A)]:
    for tag, S in [("기준", ["padi_e","cn_growth_e"]),
                   ("핵심3", ["padi_e","cn_growth_e"]+CORE3),
                   ("전체6", ["padi_e","cn_growth_e"]+USE)]:
        f = smf.ols("dd_late ~ " + " + ".join(S), data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.mid})
        bb, se, t, G = cr3(d, S, "padi_e")
        p_t = 2*(1-stats.t.cdf(abs(t), G-1))
        print(f"    {nm} {tag:5s}: β={bb:+8.2f} | CR1 SE={f.bse.padi_e:7.2f} p={f.pvalues.padi_e:.4f}"
              f" || CR3 SE={se:7.2f} t={t:+6.2f} p(t,{G-1})={p_t:.4f} "
              f"{'★비유의★' if p_t>=0.05 else ''}")
print("\n[완료]")

# -*- coding: utf-8 -*-
"""
적대적 검증 2 — (i) 자유도 보정 후 판정 변화, (ii) Bell-McCaffrey 유효자유도,
                (iii) 통제변수 간 대수적 준-공선성, (iv) 조건수를 고친 통제집합에서
                     n=23 β_PADI 가 되살아나는가 (사망이 교란인가 공선성인가)
"""
import os, sys, warnings, zlib
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
USE = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size",
       "ln_str_labor_prod", "str_wage_share", "ln_str_capital_pw"]
CORE3 = ["ln_str_capital", "str_va_ratio", "ln_str_estab_size"]
B_BOOT = 4999

P = pd.read_csv(os.path.join(FUS, "predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
C = pd.read_csv(os.path.join(FUS, "ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic": str, "k3": str, "mid": str})
D = P.merge(C[["ksic"] + USE], on="ksic", how="left")
D = D.dropna(subset=["dd_late", "padi_e", "cn_growth_e"]).reset_index(drop=True)
A = (D.groupby("k3").agg(dd_late=("dd_late","first"), padi_e=("padi_e","mean"),
                         cn_growth_e=("cn_growth_e","mean"), mid=("mid","first"))
       .reset_index().merge(D.groupby("k3")[USE].mean().reset_index(), on="k3"))
SAMPLES = [("n=39", D), ("n=23", A)]

def seed_of(*p): return zlib.crc32("|".join(map(str,p)).encode()) % (2**31)

def ols_cluster(y, X, groups):
    XtX_inv = np.linalg.pinv(X.T@X); b = XtX_inv@(X.T@y); u = y - X@b
    uq = np.unique(groups); meat = np.zeros((X.shape[1],X.shape[1]))
    for g in uq:
        m = groups==g; s = X[m].T@u[m]; meat += np.outer(s,s)
    n,k,G = len(y), X.shape[1], len(uq)
    c = (G/(G-1.0))*((n-1.0)/(n-k))
    return b, XtX_inv@meat@XtX_inv*c, u

def wild_p(d, xs, var, B=B_BOOT, seed=0):
    r = np.random.default_rng(seed); groups = d["mid"].to_numpy()
    y = d["dd_late"].to_numpy(float)
    X = np.column_stack([np.ones(len(d))]+[d[v].to_numpy(float) for v in xs])
    j = 1+xs.index(var)
    b,V,_ = ols_cluster(y,X,groups); t_obs = b[j]/np.sqrt(V[j,j])
    x0 = [v for v in xs if v!=var]
    X0 = np.column_stack([np.ones(len(d))]+[d[v].to_numpy(float) for v in x0])
    b0 = np.linalg.pinv(X0.T@X0)@(X0.T@y); fit0 = X0@b0; u0 = y-fit0
    uq = np.unique(groups); idx = {g:(groups==g) for g in uq}; ts = np.empty(B)
    for i in range(B):
        w = r.choice([-1.,1.], size=len(uq)); mult = np.ones(len(y))
        for wi,g in zip(w,uq): mult[idx[g]] = wi
        bb,Vb,_ = ols_cluster(fit0+u0*mult, X, groups)
        ts[i] = bb[j]/np.sqrt(Vb[j,j])
    return float(t_obs), float((np.abs(ts) >= abs(t_obs)-1e-12).mean())

# ── (ii) Bell-McCaffrey / Imbens-Kolesar 유효자유도 ────────────────────
def bm_dof(d, xs, var):
    """CR2 + Bell-McCaffrey 유효 자유도 K (t 분포 근사용)"""
    y = d["dd_late"].to_numpy(float); groups = d["mid"].to_numpy()
    X = np.column_stack([np.ones(len(d))]+[d[v].to_numpy(float) for v in xs])
    n,k = X.shape; uq = np.unique(groups); G = len(uq)
    XtXi = np.linalg.pinv(X.T@X)
    e = np.zeros(k); e[1+xs.index(var)] = 1.0
    a = XtXi@e                       # β_var = a'X'y
    # CR2 조정행렬
    Gmat = []
    b = XtXi@(X.T@y); u = y - X@b
    meat = np.zeros((k,k)); adj_scores=[]
    for g in uq:
        m = groups==g; Xg = X[m]
        Hg = Xg@XtXi@Xg.T
        I = np.eye(m.sum())
        # (I-Hg)^(-1/2)
        w,Vv = np.linalg.eigh(I-Hg)
        w = np.clip(w, 1e-10, None)
        Ag = Vv@np.diag(w**-0.5)@Vv.T
        adj_scores.append((m, Ag))
        meat += np.outer(Xg.T@(Ag@u[m]), Xg.T@(Ag@u[m]))
    Vcr2 = XtXi@meat@XtXi
    se_cr2 = float(np.sqrt(e@Vcr2@e))
    # BM dof: G_matrix 방식
    Gm = np.zeros((n,G))
    for i,(m,Ag) in enumerate(adj_scores):
        Ig = np.zeros((n,n)); idxs = np.where(m)[0]
        # g_i = (I_g - H_g)^{-1/2} (I - H)_{g.} a  형태를 근사 구현
        M = np.eye(n) - X@XtXi@X.T
        Gm[:,i] = (Ag@M[np.ix_(idxs,range(n))]).T@ (np.ones(len(idxs))*0)  # placeholder
    # 정확 구현: BM dof = (sum λ)^2 / sum λ^2, λ = eig(G'G), G_g = A_g (I-H)_g X a
    cols = []
    M = np.eye(n) - X@XtXi@X.T
    for (m,Ag) in adj_scores:
        idxs = np.where(m)[0]
        col = np.zeros(n)
        col[idxs] = Ag@(M[np.ix_(idxs, range(n))]@(X@a))
        cols.append(col)
    Gmat = np.column_stack(cols)
    lam = np.linalg.eigvalsh(Gmat.T@Gmat)
    lam = lam[lam>1e-12]
    dof = (lam.sum()**2)/ (lam**2).sum()
    beta = float(a@(X.T@y))
    return beta, se_cr2, float(dof)

print("="*84)
print("[G] 자유도 재판정: statsmodels 는 use_t=False(정규근사). t(G-1) / CR2+BM 로 다시 계산")
print(f"{'표본':5s} {'사양':22s} {'β':>8s} {'p(z,보고값)':>11s} {'p(t,G-1=12)':>11s} "
      f"{'CR2 SE':>8s} {'BM dof':>7s} {'p(CR2,BM)':>10s}")
specs = [("기준", ["padi_e","cn_growth_e"])] + \
        [(f"+{v}", ["padi_e","cn_growth_e",v]) for v in USE] + \
        [("핵심3", ["padi_e","cn_growth_e"]+CORE3),
         ("전체6", ["padi_e","cn_growth_e"]+USE)]
FLIP=[]
for nm,d in SAMPLES:
    for tag,xs in specs:
        f = smf.ols("dd_late ~ "+" + ".join(xs), data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.mid})
        pz = f.pvalues.padi_e; t = f.tvalues.padi_e
        pt = 2*(1-stats.t.cdf(abs(t),12))
        bb,se2,dof = bm_dof(d,xs,"padi_e")
        p2 = 2*(1-stats.t.cdf(abs(bb/se2), max(dof,1.0)))
        flag = ""
        if pz<0.05 and pt>=0.05: flag=" ←z에선 통과, t(12)에선 탈락"
        if pz<0.05 and p2>=0.05: flag+=" ←CR2/BM 탈락"
        if flag: FLIP.append((nm,tag,flag))
        print(f"{nm:5s} {tag:22s} {f.params.padi_e:+8.2f} {pz:11.4f} {pt:11.4f} "
              f"{se2:8.2f} {dof:7.2f} {p2:10.4f}{flag}")

print("\n" + "="*84)
print("[H] 통제변수 간 대수적 준-공선성: 각 구조변수를 나머지 5개에 회귀한 R²")
for nm,d in SAMPLES:
    print(f"  ── {nm}")
    for v in USE:
        oth = [x for x in USE if x!=v]
        r2 = smf.ols(f"{v} ~ "+" + ".join(oth), data=d).fit().rsquared
        print(f"     {v:<20s} R²={r2:.4f}  VIF={1/max(1e-9,1-r2):8.1f}")

print("\n" + "="*84)
print("[I] '전체6 사망' 이 교란인가 공선성인가 — 조건수를 고친 통제집합으로 재검정 (n=23)")
print("    (a) 구조6 을 주성분화(PC) 하여 직교 통제 투입, 성분 수를 1~6 으로 늘려가며")
for nm,d in SAMPLES:
    Z = d[USE].astype(float).to_numpy()
    Z = (Z - Z.mean(0))/Z.std(0, ddof=1)
    U,S,Vt = np.linalg.svd(Z, full_matrices=False)
    evr = (S**2)/ (S**2).sum()
    print(f"  ── {nm}  주성분 설명비율={np.round(evr,3)}  cond={S[0]/S[-1]:.1f}")
    dd = d.copy()
    for m in range(1,7):
        for j in range(m): dd[f"pc{j}"] = U[:,j]*S[j]
        xs = ["padi_e","cn_growth_e"]+[f"pc{j}" for j in range(m)]
        f = smf.ols("dd_late ~ "+" + ".join(xs), data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": dd.mid})
        t,wp = wild_p(dd, xs, "padi_e", seed=seed_of(nm,"pc",m))
        pt = 2*(1-stats.t.cdf(abs(f.tvalues.padi_e),12))
        print(f"     PC{m} ({evr[:m].sum()*100:5.1f}% 설명): β={f.params.padi_e:+8.2f} "
              f"p(z)={f.pvalues.padi_e:.4f} p(t12)={pt:.4f} wild-p={wp:.4f} R²={f.rsquared:.3f}")

print("\n" + "="*84)
print("[J] n=23 에서 통제변수 6개 중 3개 선택 20가지 전수 — β_PADI 분포 (사양 민감도)")
import itertools
for nm,d in SAMPLES:
    res=[]
    for S3 in itertools.combinations(USE,3):
        xs = ["padi_e","cn_growth_e"]+list(S3)
        f = smf.ols("dd_late ~ "+" + ".join(xs), data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.mid})
        pt = 2*(1-stats.t.cdf(abs(f.tvalues.padi_e),12))
        res.append((f.params.padi_e, f.pvalues.padi_e, pt, "+".join(S3)))
    res.sort()
    npass_z = sum(1 for r in res if r[1]<0.05); npass_t = sum(1 for r in res if r[2]<0.05)
    print(f"  ── {nm}: β_PADI 범위 [{res[0][0]:+.2f}, {res[-1][0]:+.2f}] "
          f"중앙값 {np.median([r[0] for r in res]):+.2f} | "
          f"p(z)<.05 통과 {npass_z}/20, p(t12)<.05 통과 {npass_t}/20")
    for r in res[-3:]:
        print(f"        최약체: β={r[0]:+8.2f} p(z)={r[1]:.4f} p(t12)={r[2]:.4f}  [{r[3]}]")
print("\n[완료]")

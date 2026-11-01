# -*- coding: utf-8 -*-
"""
적대적 검증 3 — wild cluster bootstrap-t 검정 자체의 크기(size) 점검.
  G=13 / 모수 k=9 조건에서 5% 명목수준의 실제 기각률이 얼마인가.
  '전부 5% 통과' 라는 문장이 성립하려면 검정이 5% 크기를 지켜야 한다.
"""
import os, sys, warnings, zlib
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE, "분석", "fusion")
USE = ["ln_str_capital","str_va_ratio","ln_str_estab_size",
       "ln_str_labor_prod","str_wage_share","ln_str_capital_pw"]
CORE3 = ["ln_str_capital","str_va_ratio","ln_str_estab_size"]

P = pd.read_csv(os.path.join(FUS,"predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic":str,"k3":str,"mid":str})
C = pd.read_csv(os.path.join(FUS,"ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic":str,"k3":str,"mid":str})
D = P.merge(C[["ksic"]+USE], on="ksic", how="left").dropna(
    subset=["dd_late","padi_e","cn_growth_e"]).reset_index(drop=True)
A = (D.groupby("k3").agg(dd_late=("dd_late","first"), padi_e=("padi_e","mean"),
                         cn_growth_e=("cn_growth_e","mean"), mid=("mid","first"))
       .reset_index().merge(D.groupby("k3")[USE].mean().reset_index(), on="k3"))

def design(d, xs):
    y = d["dd_late"].to_numpy(float)
    X = np.column_stack([np.ones(len(d))]+[d[v].to_numpy(float) for v in xs])
    g = d["mid"].to_numpy()
    uq = np.unique(g); Gsel = np.zeros((len(uq), len(d)))
    for i,gg in enumerate(uq): Gsel[i, g==gg] = 1.0
    return y, X, Gsel

def tstats(Y, X, Gsel, j):
    """Y: n×B. cluster-robust t 벡터 (CR1, defense_structure.py 와 동일 보정)"""
    XtXi = np.linalg.pinv(X.T@X); Q = XtXi@X.T
    B = Q@Y                      # k×B
    U = Y - X@B                  # n×B
    a = XtXi[:,j]; h = X@a       # n
    W = Gsel@(h[:,None]*U)       # G×B
    n,k = X.shape; G = Gsel.shape[0]
    c = (G/(G-1.0))*((n-1.0)/(n-k))
    V = c*(W**2).sum(0)
    return B[j]/np.sqrt(V)

def wild_p_vec(y, X, Gsel, j, B, rng):
    """관측 t 와 wild bootstrap p"""
    t_obs = tstats(y[:,None], X, Gsel, j)[0]
    keep = [i for i in range(X.shape[1]) if i != j]
    X0 = X[:,keep]
    b0 = np.linalg.pinv(X0.T@X0)@(X0.T@y); fit0 = X0@b0; u0 = y-fit0
    G = Gsel.shape[0]
    S = rng.choice([-1.,1.], size=(G,B))          # G×B
    mult = Gsel.T@S                               # n×B
    Yb = fit0[:,None] + u0[:,None]*mult
    tb = tstats(Yb, X, Gsel, j)
    return t_obs, float((np.abs(tb) >= abs(t_obs)-1e-12).mean())

SPECS = [("n=39","기준",D,["padi_e","cn_growth_e"]),
         ("n=39","핵심3",D,["padi_e","cn_growth_e"]+CORE3),
         ("n=39","전체6",D,["padi_e","cn_growth_e"]+USE),
         ("n=23","기준",A,["padi_e","cn_growth_e"]),
         ("n=23","전체6",A,["padi_e","cn_growth_e"]+USE)]

print("="*84)
print("[K] 재현 확인 (벡터화 구현 vs defense_structure.py 값)")
for nm,tag,d,xs in SPECS:
    y,X,Gs = design(d,xs)
    r = np.random.default_rng(zlib.crc32(f"{nm}|{tag}".encode())%(2**31))
    t,p = wild_p_vec(y,X,Gs,1,4999,r)
    print(f"    {nm} {tag:6s} t={t:+6.2f} wild-p={p:.4f}")

print("\n" + "="*84)
print("[L] 검정 크기(size) 시뮬레이션 — 귀무(β_PADI=0) 하에서 5%/10% 명목 기각률")
print("    귀무 DGP: 제약모형 적합 + 클러스터 단위 Rademacher 부호를 붙인 실제 잔차")
print("    (실제 이분산·클러스터 상관 구조를 그대로 보존)")
NREP, NBOOT = 400, 599
for nm,tag,d,xs in SPECS:
    y,X,Gs = design(d,xs)
    j = 1
    keep=[i for i in range(X.shape[1]) if i!=j]; X0=X[:,keep]
    b0=np.linalg.pinv(X0.T@X0)@(X0.T@y); fit0=X0@b0; u0=y-fit0
    G=Gs.shape[0]
    rng = np.random.default_rng(20260812)
    S = rng.choice([-1.,1.], size=(G,NREP)); mult = Gs.T@S
    Ysim = fit0[:,None] + u0[:,None]*mult      # 귀무 표본 NREP 개
    rej05_w=rej10_w=rej05_c=0
    from scipy import stats as st
    for r in range(NREP):
        yr = Ysim[:,r]
        t,p = wild_p_vec(yr,X,Gs,j,NBOOT,rng)
        rej05_w += p<0.05; rej10_w += p<0.10
        rej05_c += 2*(1-st.norm.cdf(abs(t))) < 0.05     # statsmodels 방식(정규근사)
    print(f"    {nm} {tag:6s} (k={X.shape[1]}, G={G}): "
          f"wild 5% 기각률={rej05_w/NREP*100:5.1f}%  10%={rej10_w/NREP*100:5.1f}%  | "
          f"정규근사 CR1 5% 기각률={rej05_c/NREP*100:5.1f}%")
print("\n[완료]  (기각률이 5% 를 크게 넘으면 '5% 통과' 라는 문장 자체가 과대주장)")

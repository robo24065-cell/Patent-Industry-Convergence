# -*- coding: utf-8 -*-
"""적대적 검증 5 — n=23 에서 통제 3개 조합 20가지 전수의 wild-p (핵심3 이 몇 등인가)
   + 통제 1~6개 모든 조합(63가지)의 생존율"""
import os, sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE,"분석","fusion")
USE = ["ln_str_capital","str_va_ratio","ln_str_estab_size",
       "ln_str_labor_prod","str_wage_share","ln_str_capital_pw"]
CORE3 = ("ln_str_capital","str_va_ratio","ln_str_estab_size")
P = pd.read_csv(os.path.join(FUS,"predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic":str,"k3":str,"mid":str})
C = pd.read_csv(os.path.join(FUS,"ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic":str,"k3":str,"mid":str})
D = P.merge(C[["ksic"]+USE], on="ksic", how="left").dropna(
    subset=["dd_late","padi_e","cn_growth_e"]).reset_index(drop=True)
A = (D.groupby("k3").agg(dd_late=("dd_late","first"), padi_e=("padi_e","mean"),
                         cn_growth_e=("cn_growth_e","mean"), mid=("mid","first"))
       .reset_index().merge(D.groupby("k3")[USE].mean().reset_index(), on="k3"))
def design(d,xs):
    y=d["dd_late"].to_numpy(float)
    X=np.column_stack([np.ones(len(d))]+[d[v].to_numpy(float) for v in xs])
    g=d["mid"].to_numpy(); uq=np.unique(g); Gs=np.zeros((len(uq),len(d)))
    for i,gg in enumerate(uq): Gs[i,g==gg]=1.
    return y,X,Gs
def tstats(Y,X,Gs,j):
    XtXi=np.linalg.pinv(X.T@X); B=(XtXi@X.T)@Y; U=Y-X@B
    a=XtXi[:,j]; W=Gs@((X@a)[:,None]*U)
    n,k=X.shape; G=Gs.shape[0]; c=(G/(G-1.))*((n-1.)/(n-k))
    return B[j]/np.sqrt(c*(W**2).sum(0))
def wild_p(d,xs,j=1,B=4999,seed=7):
    y,X,Gs=design(d,xs); rng=np.random.default_rng(seed)
    t_obs=tstats(y[:,None],X,Gs,j)[0]
    keep=[i for i in range(X.shape[1]) if i!=j]; X0=X[:,keep]
    b0=np.linalg.pinv(X0.T@X0)@(X0.T@y); fit0=X0@b0; u0=y-fit0
    S=rng.choice([-1.,1.],size=(Gs.shape[0],B))
    tb=tstats(fit0[:,None]+u0[:,None]*(Gs.T@S),X,Gs,j)
    b=np.linalg.pinv(X.T@X)@(X.T@y)
    return b[j], t_obs, float((np.abs(tb)>=abs(t_obs)-1e-12).mean())
for nm,d in [("n=39",D),("n=23",A)]:
    print("="*84); print(f"[N] {nm} — 구조통제 3개 조합 20가지 전수, wild-p 오름차순")
    res=[]
    for S3 in itertools.combinations(USE,3):
        b,t,p = wild_p(d,["padi_e","cn_growth_e"]+list(S3))
        res.append((p,b,S3))
    res.sort()
    for i,(p,b,S3) in enumerate(res,1):
        star = "  ★저자가 고른 핵심3★" if set(S3)==set(CORE3) else ""
        print(f"    {i:2d}. wild-p={p:.4f} β={b:+8.2f}  {'+'.join(S3)}{star}")
    print(f"    → wild-p<0.05 통과 {sum(1 for r in res if r[0]<0.05)}/20")
    # 전 조합(1~6개) 생존율
    tot=ok=0
    for m in range(1,7):
        for S in itertools.combinations(USE,m):
            b,t,p = wild_p(d,["padi_e","cn_growth_e"]+list(S)); tot+=1; ok+= p<0.05
    print(f"    → 통제 1~6개 전 조합 63가지 중 wild-p<0.05 생존 {ok}/{tot} ({ok/tot*100:.0f}%)")
print("\n[완료]")

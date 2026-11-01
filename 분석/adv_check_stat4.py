# -*- coding: utf-8 -*-
"""적대적 검증 4 — 경계 사양의 wild-p 몬테카를로 안정성(시드 20개)"""
import os, sys, warnings, zlib
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
FUS = os.path.join(BASE,"분석","fusion")
USE = ["ln_str_capital","str_va_ratio","ln_str_estab_size",
       "ln_str_labor_prod","str_wage_share","ln_str_capital_pw"]
P = pd.read_csv(os.path.join(FUS,"predictive_agg.csv"), encoding="utf-8-sig",
                dtype={"ksic":str,"k3":str,"mid":str})
C = pd.read_csv(os.path.join(FUS,"ctrl_structure.csv"), encoding="utf-8-sig",
                dtype={"ksic":str,"k3":str,"mid":str})
D = P.merge(C[["ksic"]+USE], on="ksic", how="left").dropna(
    subset=["dd_late","padi_e","cn_growth_e"]).reset_index(drop=True)
A = (D.groupby("k3").agg(dd_late=("dd_late","first"), padi_e=("padi_e","mean"),
                         cn_growth_e=("cn_growth_e","mean"), mid=("mid","first"))
       .reset_index().merge(D.groupby("k3")[USE].mean().reset_index(), on="k3"))
def design(d, xs, yname="dd_late"):
    y = d[yname].to_numpy(float)
    X = np.column_stack([np.ones(len(d))]+[d[v].to_numpy(float) for v in xs])
    g = d["mid"].to_numpy(); uq=np.unique(g)
    Gs = np.zeros((len(uq),len(d)))
    for i,gg in enumerate(uq): Gs[i, g==gg]=1.
    return y,X,Gs
def tstats(Y,X,Gs,j):
    XtXi=np.linalg.pinv(X.T@X); Q=XtXi@X.T; B=Q@Y; U=Y-X@B
    a=XtXi[:,j]; h=X@a; W=Gs@(h[:,None]*U)
    n,k=X.shape; G=Gs.shape[0]; c=(G/(G-1.))*((n-1.)/(n-k))
    return B[j]/np.sqrt(c*(W**2).sum(0))
def wild_p(y,X,Gs,j,B,rng):
    t_obs=tstats(y[:,None],X,Gs,j)[0]
    keep=[i for i in range(X.shape[1]) if i!=j]; X0=X[:,keep]
    b0=np.linalg.pinv(X0.T@X0)@(X0.T@y); fit0=X0@b0; u0=y-fit0
    S=rng.choice([-1.,1.],size=(Gs.shape[0],B)); mult=Gs.T@S
    tb=tstats(fit0[:,None]+u0[:,None]*mult,X,Gs,j)
    return t_obs, float((np.abs(tb)>=abs(t_obs)-1e-12).mean())

CASES=[("n=39 +str_wage_share(보고 0.0410)", D, ["padi_e","cn_growth_e","str_wage_share"],"padi_e"),
       ("n=39 전체6      (보고 0.0254)", D, ["padi_e","cn_growth_e"]+USE,"padi_e"),
       ("n=23 +str_va_ratio(보고 0.0644)", A, ["padi_e","cn_growth_e","str_va_ratio"],"padi_e"),
       ("n=23 핵심3      (보고 0.0400)", A, ["padi_e","cn_growth_e","ln_str_capital","str_va_ratio","ln_str_estab_size"],"padi_e"),
       ("n=23 FWL-b      (보고 0.0374)", A, ["p_r2","cn_growth_e"],"p_r2")]
print("="*84)
print("[M] 경계 사양 wild-p 의 시드 민감도 (B=4999, 시드 20개)")
for tag,d,xs,var in CASES:
    d=d.copy()
    if "p_r2" in xs:
        d["p_r2"]=smf.ols("padi_e ~ "+" + ".join(USE),data=d).fit().resid
    y,X,Gs=design(d,xs); j=1+xs.index(var)
    ps=[]
    for s in range(20):
        _,p=wild_p(y,X,Gs,j,4999,np.random.default_rng(1000+s)); ps.append(p)
    ps=np.array(ps)
    print(f"    {tag:34s} 평균={ps.mean():.4f} 최소={ps.min():.4f} 최대={ps.max():.4f} "
          f"sd={ps.std():.4f} | 5%미만 시드 {int((ps<0.05).sum())}/20")
print("\n[완료]")

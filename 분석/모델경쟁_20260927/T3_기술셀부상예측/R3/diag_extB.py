"""B단계 신규 외부 피처 단변량 진단(개발 원점 2010–2018, 공개연도 변형 레이블만)."""
import os, sys; os.environ['T3R3_VAR']='pub'; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import numpy as np, pandas as pd, r3core as C
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.argv=['x','E32']; exec(open(os.path.join(HERE,'exps3b.py'),encoding='utf-8').read().split("if __name__")[0])
D=prep(C.get_D()); D=D[D.t0.isin(C.DEV_T0)]
fs=ROS+OAC+KI+CU+['ix_sci_fx','ix_oa_gap']
rows=[]
for f in fs:
    a=[];r=[];pc=[]
    for t,g in D.groupby('t0'):
        m=g[f].notna()
        if m.sum()<100 or g.yb[m].nunique()<2: continue
        a.append(roc_auc_score(g.yb[m],g[f][m])); r.append(spearmanr(g[f][m],g.y[m]).statistic)
        # 편상관: kr_mom, fx_mom, kr_lvl 잔차
        X=np.c_[np.ones(m.sum()),g.loc[m,['kr_mom','fx_mom','kr_lvl','tot_lvl']].fillna(0).values]
        rk=lambda v: pd.Series(v).rank().values
        res=lambda v: v-X@np.linalg.lstsq(X,v,rcond=None)[0]
        pc.append(np.corrcoef(res(rk(g[f][m].values)),res(rk(g.y[m].values)))[0,1])
    rows.append(dict(f=f,nonmiss=round(D[f].notna().mean(),3),auc=round(np.mean(a),4),rho=round(np.mean(r),4),pcor=round(np.mean(pc),4),pcor_pos=int(np.sum(np.array(pc)>0)),n_org=len(pc)))
out=pd.DataFrame(rows); print(out.to_string()); out.to_csv(os.path.join(HERE,'extB_univariate.csv'),index=False,encoding='utf-8-sig')

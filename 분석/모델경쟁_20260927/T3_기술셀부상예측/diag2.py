import numpy as np, pandas as pd, harness as H, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
D=H.data(); V=D[D.t0.isin([2011,2013,2015])]
P=pd.read_parquet('preds/v17.parquet').merge(V[['cell','t0','kr_W','tot_W','ksic','mg','kr_fut','kr_prio_for_sh']],on=['cell','t0'])
f=lambda g: pd.Series(dict(n=len(g),evt=g.yb.mean(),auc=roc_auc_score(g.yb,g.p) if 0<g.yb.sum()<len(g) else np.nan))
P['szb']=pd.cut(P.kr_W,[-1,0,5,20,100,1e9],labels=['0','1-5','6-20','21-100','>100'])
P['totb']=pd.qcut(P.tot_W,4)
print(P.groupby('szb',observed=True).apply(f).round(3)); print(P.groupby('totb',observed=True).apply(f).round(3))
# 대분류(KSIC 2자리)
P['k2']=P.ksic.str[:2]; print(P.groupby('k2').apply(f).round(3).sort_values('auc'))
P['sec']=P.mg.str[0]; print(P.groupby('sec').apply(f).round(3))
# 상위 10% 선정 셀 중 실제 KR 증가량
P['rk']=P.groupby('t0').p.rank(ascending=False,pct=True)
top=P[P.rk<=0.1]
print('top10 picks: mean kr_W %.1f, mean kr_fut %.1f, hit %.3f, abs gain median %.1f'%(top.kr_W.mean(),top.kr_fut.mean(),top.yb.mean(),(top.kr_fut-top.kr_W).median()))
ev=P[P.yb==1]; print('events: kr_W median',ev.kr_W.median(),'kr_fut median',ev.kr_fut.median(),'share kr_W==0',(ev.kr_W==0).mean())
# mg 군집: 같은 mg 여러 KSIC
print('cells per mg in validation (mean)', P.groupby(['t0','mg']).size().mean(), 'n mg', P.mg.nunique())

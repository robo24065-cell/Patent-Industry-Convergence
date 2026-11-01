import numpy as np, pandas as pd, harness as H
from sklearn.metrics import roc_auc_score
D=H.data(); V=D[D.t0.isin([2011,2013,2015])]
for v in ['v0','v2','v5']:
    P=pd.read_parquet(f'preds/{v}.parquet').merge(V[['cell','t0','kr_W','tot_W','ksic','kr_share','kr_lvl','kr_mom','US_mom','CN_mom']],on=['cell','t0'])
    P['szb']=pd.cut(P.kr_W,[-1,0,5,20,100,1e9],labels=['0','1-5','6-20','21-100','>100'])
    print('==',v)
    print(P.groupby('szb',observed=True).apply(lambda g: pd.Series(dict(n=len(g),evt_rate=g.yb.mean(),auc=roc_auc_score(g.yb,g.p) if 0<g.yb.sum()<len(g) else np.nan,mean_p=g.p.mean()))).round(3))
# single-feature AUCs pooled per t0 mean
feats=[c for c in V.columns if c not in ('cell','ksic','mg','t0','y','yb','kr_fut')]
rows=[]
for f in feats:
    a=[]
    for t,g in V.groupby('t0'):
        x=g[f].fillna(g[f].median())
        a.append(roc_auc_score(g.yb,x))
    rows.append((f,np.mean(a)))
S=pd.DataFrame(rows,columns=['f','auc']).assign(dev=lambda d:(d.auc-0.5).abs()).sort_values('dev',ascending=False)
print(S.head(30).to_string())
P=pd.read_parquet('preds/v5.parquet').merge(V[['cell','t0','ksic']],on=['cell','t0'])
k=P.groupby('ksic').apply(lambda g: pd.Series(dict(n=len(g),evt=g.yb.mean(),auc=roc_auc_score(g.yb,g.p) if 0<g.yb.sum()<len(g) else np.nan))).sort_values('auc')
print(k.head(8).round(3)); print(k.tail(5).round(3))

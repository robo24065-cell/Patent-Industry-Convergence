import numpy as np, pandas as pd, t4lib as T
D=T.load(); pt=D['pt']; rt=D['rt']; pay=D['pay']
L='2006-01-01'
pids,pth,l=T.risk_set(L); ya=T.label_asof(L,pids,'2014-01-01')
u=pids[np.isnan(ya)]
pthC,lC=T.paid_through('2014-01-01')
print('unknown n',len(u))
print('paid_year asof C', pd.Series(lC[u]).value_counts().head(8).to_dict())
print('exp year', pd.Series(pd.DatetimeIndex(pt.exp.values[u]).year).value_counts().head(8).to_dict())
r=rt[rt.pid.isin(u)&(rt.cd<=np.datetime64('2014-01-01'))]
print(r.groupby(['rgstCsName','rgstCsReason'],dropna=False).size().sort_values(ascending=False).head(15))
print(pt.iloc[u][['rgst_no','rgstDate','applDate','cndrtExptnDate','atype']].head(8))

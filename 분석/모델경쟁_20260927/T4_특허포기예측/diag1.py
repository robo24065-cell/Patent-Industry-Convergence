import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from versions import *
ev=get_frame(EVAL_L,None); tr=get_frame(train_landmarks(C_VAL),C_VAL)
p=pd.read_parquet('preds/pred_v4.parquet'); ev['p4']=p.p.values
p0=pd.read_parquet('preds/pred_v0.parquet'); ev['p0']=p0.p.values
ev['ageb']=pd.cut(ev.age,[0,1,2,3,4,5,6,7,8,10,12,20])
tr['ageb']=pd.cut(tr.age,[0,1,2,3,4,5,6,7,8,10,12,20])
def seg(col):
    r=ev.groupby(col,observed=True).apply(lambda d: pd.Series(dict(n=len(d),rate=d.y.mean(),auc_v0=roc_auc_score(d.y,d.p0) if d.y.nunique()>1 else np.nan,auc_v4=roc_auc_score(d.y,d.p4) if d.y.nunique()>1 else np.nan, mean_p4=d.p4.mean())))
    print(r.round(3).to_string())
seg('ageb'); print('train age rate', tr.groupby('ageb',observed=True).y.agg(['size','mean']).round(3).to_string())
seg('atype'); seg('next_annl')
print('train next_annl', tr.groupby('next_annl').y.agg(['size','mean']).round(3).head(15).to_string())
# ceiling: grouped CV within eval (diagnostic only)
gkf=GroupKFold(5); oof=np.zeros(len(ev))
num=BASE_NUM+PAYSTRUCT+PAYBEH+HIST; cat=BASE_CAT+HIST_CAT
for k,(a,b) in enumerate(gkf.split(ev,groups=ev.acd)):
    if k>1: break
    oof[b]=xgb_fit(ev.iloc[a],ev.iloc[b],num,cat,rounds=300)
    print('within-period CV fold',k,roc_auc_score(ev.y.iloc[b],oof[b]))

import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
from run import *
ev=get_frame(EVAL_L,None)
vs=sys.argv[1:]
for v in vs: ev[v]=pd.read_parquet(f'preds/pred_{v}.parquet').p.values
ev['ageb']=pd.cut(ev.age,[0,1,2,3,4,6,8,10,20]); ev['ndw']=ev.n_due_win.clip(0,4)
def a(d,c): 
    return roc_auc_score(d.y,d[c]) if d.y.nunique()>1 else np.nan
for col in ['ageb','ndw','atype','L']:
    r=ev.groupby(col,observed=True).apply(lambda d: pd.Series({'n':len(d),'rate':d.y.mean(),**{f'auc_{v}':a(d,v) for v in vs},**{f'mp_{v}':d[v].mean() for v in vs}}))
    print(r.round(3).to_string())

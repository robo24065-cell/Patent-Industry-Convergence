import sys,numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from common import *; from sklearn.metrics import roc_auc_score
tr=load_panel(range(2006,2012)); va=load_panel(VAL_L)
num=[c for c in tr.columns if tr[c].dtype.kind in 'if' and not c.startswith('y_') and c not in ('L',)]
out=[]
for c in num:
    a=[]
    for d in (tr,va):
        x=d[c].fillna(-999); a.append(roc_auc_score(d.y_MAIN,x))
    out.append((c,round(a[0],3),round(a[1],3),round(va[c].isna().mean(),3)))
print(pd.DataFrame(out,columns=['feat','auc_tr','auc_va','na_va']).sort_values('auc_va').to_string())
for c in ['orig_type','cur_type','applTpcd','lastDspst']:
    print(va.groupby(c).y_MAIN.agg(['size','mean']).round(4).T.to_string())
    print(tr.groupby(c).y_MAIN.agg(['size','mean']).round(4).T.to_string())
print(va.groupby('ksic').y_MAIN.agg(['size','mean']).sort_values('mean').round(4).to_string())
# bundle concentration
pos=va[va.y_MAIN==1]; print('val pos by cur top', pos.cur.value_counts().head(15).to_string())
print('train pos by cur top', tr[tr.y_MAIN==1].cur.value_counts().head(10).to_string())

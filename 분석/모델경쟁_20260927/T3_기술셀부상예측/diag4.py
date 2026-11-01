# IPC 재분류(스냅샷) 누수 점검: 기준시점 이후 신설된 서브클래스 셀 비중과 제외 시 AUC
import numpy as np, pandas as pd, harness as H, t3lib as L
from sklearn.metrics import roc_auc_score
NEWSUB={'G16B','G16C','G16H','G16Y','G16Z','B64U','H10B','H10K','H10N','G06V','B33Y'}
out=[]
for v in ['v0','v22']:
    P=pd.read_parquet(f'preds/{v}.parquet'); P['sub']=P.cell.str.split('|').str[1].str[:4]
    for t,g in P.groupby('t0'):
        m=g['sub'].isin(NEWSUB)
        out.append(dict(v=v,t0=t,n_new=int(m.sum()),evt_new=int(g.yb[m].sum()),evt_rate_new=g.yb[m].mean() if m.any() else np.nan,
                        auc_all=roc_auc_score(g.yb,g.p),auc_excl=roc_auc_score(g.yb[~m],g.p[~m])))
R=pd.DataFrame(out); print(R.round(4).to_string()); print(R.groupby('v')[['auc_all','auc_excl']].mean().round(4))
P=pd.read_parquet('preds/v22.parquet'); P['sub']=P.cell.str.split('|').str[1].str[:4]
print(P[P['sub'].isin(NEWSUB)].groupby(['t0','sub']).agg(n=('yb','size'),evt=('yb','sum'),mean_p=('p','mean')))

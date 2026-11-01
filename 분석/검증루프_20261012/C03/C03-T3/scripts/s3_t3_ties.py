import sys, json, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
M=sys.argv[1]; W=sys.argv[2]
df=pd.read_parquet(M+'/T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet')
print(df.columns.tolist(), len(df), int(df['yb'].sum()))
def tie_range(s,y,k):
    s=np.asarray(s,float); y=np.asarray(y,int)
    thr=np.sort(s)[::-1][k-1]
    above=s>thr; tie=s==thr
    na=above.sum(); need=k-na; nt=tie.sum(); tp=y[tie].sum(); pa=y[above].sum()
    lo=pa+max(0,need-(nt-tp)); hi=pa+min(need,tp); ex=pa+need*tp/nt
    return dict(k=k,n_above=int(na),n_tie=int(nt),tie_pos=int(tp),need=int(need),pos_above=int(pa),min=int(lo),max=int(hi),expected=round(float(ex),2),
                min_pct=round(100*lo/k,2),max_pct=round(100*hi/k,2),exp_pct=round(100*ex/k,2))
out={}
for col in ['p','p_base']:
    out[col]={str(k):tie_range(df[col],df['yb'],k) for k in (100,504)}
    # pandas sort stable by index for reference
    o=df.sort_values(col,ascending=False,kind='mergesort')
    out[col]['stable_top504']=int(o['yb'].head(504).sum()); out[col]['stable_top100']=int(o['yb'].head(100).sum())
from sklearn.metrics import roc_auc_score
out['auc_p']=roc_auc_score(df['yb'],df['p']); out['auc_base']=roc_auc_score(df['yb'],df['p_base'])
json.dump(out,open(W+'/t3_baseline_ties_recheck.json','w'),indent=1)
print(json.dumps(out,indent=1))

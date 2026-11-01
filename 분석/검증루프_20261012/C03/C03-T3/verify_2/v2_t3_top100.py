import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
P=sys.argv[1]; OUT=sys.argv[2]
df=pd.read_parquet(P)
print(df.shape, df.columns.tolist())
print(df.head(3).to_string())
y=df['yb'].astype(int).values
res={}
for col in ['p','p_base']:
    s=df[col].astype(float).values
    r={}
    order=np.sort(s)[::-1]
    for k in (100,504):
        thr=order[k-1]
        above=s>thr; tie=s==thr
        # near-ties check (floating noise)
        near=np.isclose(s,thr,rtol=0,atol=1e-12)
        na=int(above.sum()); nt=int(tie.sum()); tp=int(y[tie].sum()); pa=int(y[above].sum()); need=k-na
        lo=pa+max(0,need-(nt-tp)); hi=pa+min(need,tp)
        # hypergeometric expectation
        ex=pa+need*tp/nt
        # orders
        qs=int(y[np.argsort(-s,kind='quicksort')[:k]].sum())
        st=int(y[np.argsort(-s,kind='stable')[:k]].sum())
        pd_def=int(df.sort_values(col,ascending=False)['yb'].head(k).sum())
        pd_nl=int(df.nlargest(k,col)['yb'].sum())
        rng=np.random.default_rng(0); sims=[]
        for _ in range(20000):
            sel=rng.choice(np.where(tie)[0],need,replace=False); sims.append(pa+y[sel].sum())
        sims=np.array(sims)
        r[str(k)]=dict(thr=float(thr),n_above=na,n_tie=nt,n_near=int(near.sum()),tie_pos=tp,pos_above=pa,need=need,min=int(lo),max=int(hi),
            expected=round(ex,3),sim_mean=round(float(sims.mean()),3),sim_p2_5=float(np.percentile(sims,2.5)),sim_p97_5=float(np.percentile(sims,97.5)),
            numpy_quicksort=qs,numpy_stable=st,pandas_sort_default=pd_def,pandas_nlargest=pd_nl)
    r['n_unique']=int(pd.Series(s).nunique())
    res[col]=r
res['n']=len(df); res['n_pos']=int(y.sum())
from sklearn.metrics import roc_auc_score
res['auc_p']=roc_auc_score(y,df['p']); res['auc_base']=roc_auc_score(y,df['p_base'])
json.dump(res,open(OUT,'w',encoding='utf-8'),indent=1,ensure_ascii=False)
print(json.dumps(res,indent=1,ensure_ascii=False))

import numpy as np, pandas as pd, xgboost as xgb
from versions import *
tr = pd.concat([HZ.train_decisions(L, C_VAL) for L in ANNUAL], ignore_index=True)
ev = get_frame(EVAL_L, None)
decs=[]
for L in EVAL_L:
    e = ev[ev.L == L].reset_index(); d = HZ.eval_decisions(e); d['row'] = e['index'].values[d.row.values]; decs.append(d)
dec = pd.concat(decs, ignore_index=True)
A, B = add_ratio(tr), add_ratio(dec)
cols = H12N + F6C
h, bst = xgb_fit(A, B, H12N, F6C, weight=recency_w(tr), rounds=600, return_model=True)
def M(d):
    X = d[cols].copy()
    for c in F6C: X[c] = pd.Categorical(X[c].astype(str), categories=sorted(pd.concat([A[c], B[c]]).astype(str).unique()))
    return X
def contrib(d, n=20000):
    s = d.sample(min(n, len(d)), random_state=0)
    c = bst.predict(xgb.DMatrix(M(s), enable_categorical=True), pred_contribs=True)
    return pd.Series(c.mean(0), index=cols + ['bias'])
ptr = bst.predict(xgb.DMatrix(M(A), enable_categorical=True))
A['h'] = ptr
print('train in-sample by k'); print(A.groupby(A.k.clip(upper=13)).agg(n=('y','size'), true=('y','mean'), pred=('h','mean')).round(3).to_string())
ct = contrib(A[(A.k >= 9) & (A.k <= 11)]); ce = contrib(B[(B.k >= 9) & (B.k <= 11)])
df = pd.DataFrame({'train': ct, 'eval': ce}); df['diff'] = df['eval'] - df['train']
print(df.reindex(df['diff'].abs().sort_values(ascending=False).index).head(15).round(3).to_string())

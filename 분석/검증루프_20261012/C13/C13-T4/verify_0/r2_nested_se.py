import sys, json, numpy as np, pyarrow.parquet as pq
from scipy.stats import norm
p=sys.argv[1]
d=pq.read_table(p).to_pandas()
print(len(d), d.y.sum(), d.u.nunique(), d.modal_prov.nunique(), d.modal_prov.isna().sum())
y=d.y.values
def wauc(s,w):
    o=np.argsort(s,kind='mergesort'); s2=s[o]; y2=y[o]; w2=w[o]
    # ties: average rank via groups
    wp=w2*y2; wn=w2*(1-y2)
    # cumulative negatives weight below each score group
    uniq, idx, inv = np.unique(s2, return_index=True, return_inverse=True)
    gp=np.bincount(inv,weights=wp); gn=np.bincount(inv,weights=wn)
    cn=np.cumsum(gn)-gn
    num=(gp*(cn+0.5*gn)).sum()
    return num/(gp.sum()*gn.sum())
w1=np.ones(len(y))
A={k:wauc(d['p_'+k].values,w1) for k in ['base','prov','free','hsc']}
print('AUC',A)
dA=A['prov']-max(A['free'],A['hsc'])
print('dAUC prov-best', dA)
uid=d.u.astype('category').cat.codes.values; nu=uid.max()+1
kid=d.modal_prov.astype('category').cat.codes.values; nk=kid.max()+1
rng=np.random.default_rng(99)
B=400
res={'two':[], 'firm':[], 'ksic':[], 'two_free':[]}
sp=d.p_prov.values; sh=d.p_hsc.values; sf=d.p_free.values
for b in range(B):
    wu=rng.poisson(1,nu)[uid].astype(float); wk=rng.poisson(1,nk)[kid].astype(float)
    for name,w in [('two',wu*wk),('firm',wu),('ksic',wk)]:
        res[name].append(wauc(sp,w)-wauc(sh,w))
    res['two_free'].append(wauc(sp,wu*wk)-max(wauc(sh,wu*wk),wauc(sf,wu*wk)))
out={}
scale=np.sqrt(2105/1981.51)
for k,v in res.items():
    se=np.std(v,ddof=1); sel=se*scale
    out[k]=dict(se_dev=se, se_lock=sel, power=float(norm.cdf(0.002/sel-1.96)), mean=float(np.mean(v)))
print(json.dumps(out,indent=1))

from models import *
def base_fp(tr,te):
    m=LogisticRegression(C=1e6,max_iter=1000).fit(tr[['kr_mom']].values,tr.yb.values); return m.predict_proba(te[['kr_mom']].values)[:,1]
rows=[]
for mt in [20,100]:
    D=enrich(H.data(mt))
    R0,_=H.rolling(base_fp,D); R,_=H.rolling(best_fp(),D)
    a0=H.summarize(R0)['auc']; a=H.summarize(R)['auc']
    rows.append(dict(min_total=mt,n='/'.join(map(str,R.n)),evt='/'.join(map(str,R.n_evt)),base=a0,v22=a,skill=(a-a0)/(1-a0),p10=H.summarize(R)['p_at10']))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('sensitivity_min_total.csv',index=False)

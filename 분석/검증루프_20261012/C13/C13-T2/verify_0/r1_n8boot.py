# 독립 구현: N8 군집 부트스트랩(가중 AUC, 동점 0.5). 작업 코드/n8lib 미사용.
import sys, json, numpy as np, pandas as pd
base=sys.argv[1]; bundles=[int(x) for x in sys.argv[2].split(',')]
def prep(p):
    u,inv=np.unique(p,return_inverse=True); return inv,len(u)
def wauc(y,inv,K,w):
    wp=np.bincount(inv,w*y,K); wn=np.bincount(inv,w*(1-y),K)
    below=np.cumsum(wn)-wn
    return (wp*(below+0.5*wn)).sum()/(wp.sum()*wn.sum())
out={}
for b in bundles:
    d=pd.read_parquet(f'{base}/cache/n8/preds_b{b}.parquet')
    y=d.y.values.astype(float)
    codes,uniq=pd.factorize(d.g.values); G=len(uniq)
    P={k:prep(d['p_'+k].values) for k in ['champ','M0','SB']}
    one=np.ones(len(y))
    A={k:wauc(y,*P[k],one) for k in P}
    res={'auc':A,'skill':(A['champ']-A['M0'])/(1-A['M0']),'skill_sb':(A['champ']-A['SB'])/(1-A['SB']),'G':G}
    for lab,seed,conv in [('same2022',2022,'multi'),('same2023',2023,'multi'),('alt7',7,'int'),('alt8',8,'int')]:
        rng=np.random.default_rng(seed); S=[];T=[]
        for _ in range(1000):
            if conv=='multi': w=rng.multinomial(G,np.full(G,1.0/G))[codes].astype(float)
            else: w=np.bincount(rng.integers(0,G,G),minlength=G)[codes].astype(float)
            if (w*y).sum()==0: continue
            a=wauc(y,*P['champ'],w); a0=wauc(y,*P['M0'],w); asb=wauc(y,*P['SB'],w)
            S.append((a-a0)/(1-a0)); T.append((a-asb)/(1-asb))
        res[lab]={'M0':list(np.percentile(S,[2.5,97.5])),'SB':list(np.percentile(T,[2.5,97.5]))}
    out[b]=res; print(b,json.dumps(res),flush=True)
json.dump(out,open(f'{base}/verify_0/r1_n8boot.json','w'),indent=1)

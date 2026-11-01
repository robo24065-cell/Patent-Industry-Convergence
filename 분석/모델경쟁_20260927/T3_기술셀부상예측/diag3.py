from models import *
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
D=enrich(H.data())
# (1) 부트스트랩: mg 군집 재표집(원점별), v22 vs v0 / v17 / v13
Pv={v:pd.read_parquet(f'preds/{v}.parquet') for v in ['v0','v13','v17','v22']}
V=Pv['v22'][['cell','t0','yb']].copy(); V['mg']=V.cell.str.split('|').str[1]
for v in Pv: assert (Pv[v].cell.values==V.cell.values).all(); V[v]=Pv[v].p.values
B=1000; res={k:[] for k in ['v22','v0','d_v22_v0','d_v22_v17','d_v22_v13']}
groups={t:{m:np.where((V.t0==t)&(V.mg==m))[0] for m in V[V.t0==t].mg.unique()} for t in L.VAL_T0}
for b in range(B):
    a={v:[] for v in ['v0','v13','v17','v22']}
    for t in L.VAL_T0:
        ms=list(groups[t].keys()); pick=rng.choice(len(ms),len(ms)); ii=np.concatenate([groups[t][ms[i]] for i in pick])
        y=V.yb.values[ii]
        for v in a: a[v].append(roc_auc_score(y,V[v].values[ii]))
    m={v:np.mean(a[v]) for v in a}
    res['v22'].append(m['v22']); res['v0'].append(m['v0']); res['d_v22_v0'].append(m['v22']-m['v0'])
    res['d_v22_v17'].append(m['v22']-m['v17']); res['d_v22_v13'].append(m['v22']-m['v13'])
for k,x in res.items(): print('boot',k,'mean %.4f  95%%CI [%.4f, %.4f]'%(np.mean(x),*np.percentile(x,[2.5,97.5])))
sk=(np.array(res['v22'])-np.array(res['v0']))/(1-np.array(res['v0'])); print('skill CI [%.3f, %.3f]'%tuple(np.percentile(sk,[2.5,97.5])))
# (2) 잡음 천장: NB 모형이 참이라 가정하고 시뮬레이션한 레이블에 대한 자기 꼬리확률 AUC
for v in L.VAL_T0:
    tr=D[D.t0<=v-3]; te=D[D.t0==v]
    s,mu=nbfp(ALL2,return_mu=True)(tr,te)
    d=xgb.DMatrix(tr[ALL2].values,label=tr.kr_fut.values)
    a=0.212 if v==2013 else (0.191 if v==2011 else 0.236)
    r=1/a; p=r/(r+mu); aucs=[]
    for k in range(50):
        F=rng.negative_binomial(r,p); y=np.log((F+1)/(te.kr_W.values+1)); yb=L.breakout(y,tie=F-te.kr_W.values)
        aucs.append(roc_auc_score(yb,s))
    print('ceiling t0',v,'simulated AUC mean %.3f sd %.3f'%(np.mean(aucs),np.std(aucs)))

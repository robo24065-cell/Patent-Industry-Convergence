import numpy as np, pandas as pd, t3lib as L
P=L.load(); A=P['A']
print('cells', len(P['cells']), 'ksic', len(set(P['ksic'])), 'mg', len(set(P['mg'])))
for t0 in L.VAL_T0:
    y=L.labels(P,t0)
    for m in [1,10,20,30,50,100,200]:
        e=L.eligible(P,t0,m); ye=y[e]; kr=L.wsum(A['n'][:,0,:],t0,-2,0)[e]
        yb=L.breakout(ye)
        print(t0,'min',m,'elig',e.sum(),'uniq_y',len(np.unique(ye)),'evt',yb.sum(),'thr %.3f'%np.quantile(ye,.9),'mean %.3f sd %.3f'%(ye.mean(),ye.std()),
              'KRpast0 %.3f'%(kr==0).mean(), 'evt_KRpast0 %.3f'%(kr[yb==1]==0).mean(), 'evt_KRpast<=5 %.3f'%(kr[yb==1]<=5).mean())

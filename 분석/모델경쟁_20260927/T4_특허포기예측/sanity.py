import time, numpy as np, pandas as pd, t4lib as T
t=time.time(); D=T.load(); print('load',round(time.time()-t,1))
pt=D['pt']; print('patents',len(pt)); print('lapse events kept (E<2019)',len(D['lap']))
print('E year dist', pd.Series(pd.DatetimeIndex(D['lap'].E).year).value_counts().sort_index().to_dict())
for L in ['2006-01-01','2009-07-01','2014-01-01','2015-07-01']:
    t=time.time(); pids,pth,l=T.risk_set(L)
    y=T.label_full(L,pids)
    ya=T.label_asof(L,pids,'2014-01-01') if L<'2010' else None
    s=f'{L}: risk {len(pids)} events {y.sum()} rate {y.mean():.4f}'
    if ya is not None: s+=f' | asof2014: known {np.isfinite(ya).sum()} pos {np.nansum(ya):.0f} unknown {np.isnan(ya).sum()} agree {(ya[np.isfinite(ya)]==y[np.isfinite(ya)]).mean():.4f}'
    print(s, round(time.time()-t,1))

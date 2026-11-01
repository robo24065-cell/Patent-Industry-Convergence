import numpy as np, pandas as pd
from versions import *
m, pl, tr, dec, ev, p, h = hazard_run('tmp_v9', NOCAL + DEC, F6C, ANNUAL, ret=True, rounds=300)
lap=T.load()['lap']; pt=T.load()['pt']
key=set(zip(lap.pid.values, lap.E.values.astype('datetime64[D]')))
pid=ev.pid.values[dec.row.values]
dd=T.add_years(pt.rd.values[pid], dec.k.values.astype(int)-1)
dec['yd']=np.array([(a,b) in key for a,b in zip(pid,dd)]).astype(int); dec['h']=h
# condition on survival: decision reached = no lapse at earlier decision in window
dec=dec.sort_values(['row','j']); dec['prev']=dec.groupby('row').yd.cumsum()-dec.yd
dr=dec[dec.prev==0]
print('eval decisions (reached): by k'); print(dr.groupby(dr.k.clip(upper=13)).agg(n=('yd','size'),true=('yd','mean'),pred=('h','mean')).round(3).to_string())
print('by j'); print(dr.groupby('j').agg(n=('yd','size'),true=('yd','mean'),pred=('h','mean')).round(3).to_string())
print('train by j'); print(tr.groupby('j').agg(n=('y','size'),true=('y','mean')).round(3).to_string())
cols=['lead','age_d','n_renew','paid_ahead','to_expiry','a_lapse_rate','port_n','n_late','mean_early','yrs_since_pay','fee_rel','last_amt_per_yr','max_span','last_span']
a=tr[(tr.k>=8)&(tr.k<=10)][cols].mean(); b=dr[(dr.k>=8)&(dr.k<=10)][cols].mean()
print(pd.DataFrame({'train_k8_10':a,'eval_k8_10':b}).round(3).to_string())

import numpy as np, pandas as pd
from versions import *
tr = pd.concat([HZ.train_decisions(L, C_VAL) for L in ANNUAL], ignore_index=True)
print(pd.crosstab(tr.rgst_year, tr.k.clip(upper=12), values=tr.y, aggfunc='mean').round(3).to_string())
print(pd.crosstab(tr.rgst_year, tr.k.clip(upper=12)).to_string())
# eval truth at decision level (validation only)
ev=get_frame(EVAL_L,None); lap=T.load()['lap']
e=ev[ev.L=='2014-01-01'].reset_index(drop=True); d=HZ.eval_decisions(e)
key=set(zip(lap.pid.values, lap.E.values.astype('datetime64[D]')))
d['yd']=[ (p,dd) in key for p,dd in zip(e.pid.values[d.row.values], d.age_d.values*0+0)] if False else 0
dd=T.add_years(T.load()['pt'].rd.values[e.pid.values[d.row.values]], d.k.values.astype(int)-1)
d['yd']=[(p,x) in key for p,x in zip(e.pid.values[d.row.values], dd)]
print('eval L=2014-01 decision-level hazard (unconditional on survival) by rgst_year x k')
print(pd.crosstab(d.rgst_year, d.k.clip(upper=12), values=d.yd, aggfunc='mean').round(3).to_string())

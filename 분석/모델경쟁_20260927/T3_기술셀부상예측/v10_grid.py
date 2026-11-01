from models import *
import time
D=H.data()
res=[]
for md in [2,3,4,6]:
  for rounds,eta in [(150,0.05),(300,0.05),(600,0.03),(300,0.02)]:
    for mcw in [5,20]:
        t=time.time()
        R,P=H.rolling(xgbc(ALL,dict(max_depth=md,eta=eta,min_child_weight=mcw),rounds=rounds),D)
        s=H.summarize(R); res.append((md,rounds,eta,mcw,s['auc'],s['auc_by_t0'],s['p_at10']))
        print(res[-1], round(time.time()-t,1), flush=True)
pd.DataFrame(res,columns=['depth','rounds','eta','mcw','auc','by_t0','p10']).to_csv('grid_v10.csv',index=False)

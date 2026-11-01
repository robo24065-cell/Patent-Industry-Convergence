import sys,numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from common import *; from evalkit import *
va=load_panel(VAL_L).merge(pd.read_parquet('cache/pred_v5.parquet'),on=['rgst_no','L'])
p=va.p.values
pos=va[va.y_MAIN==1]; bk=pos.groupby(['L','cur','y_date']).rgst_no.transform('size')
va['bundle']=0; va.loc[pos.index,'bundle']=bk.values
for lo,hi in [(1,1),(2,5),(6,50),(51,99999)]:
    k=(va.y_MAIN==0)|((va.bundle>=lo)&(va.bundle<=hi))
    print('bundle',lo,hi,'pos',int(((va.bundle>=lo)&(va.bundle<=hi)).sum()),'AUC',round(roc_auc_score(va.y_MAIN[k],p[k]),4))
va['port_b']=pd.cut(va.own_port,[0,1,5,20,100,1000,1e9])
va['age_b']=pd.cut(va.age,[0,2,4,6,8,10,30])
for c in ['cur_type','port_b','age_b','L']: print(seg_report(va,p,c).to_string())
r=seg_report(va,p,'ksic'); print(r.sort_values('auc').to_string())
# 기여 큰 번들 양도인: 예측평균 vs 실제
g=va[va.bundle>50].groupby(['cur']).agg(n=('rgst_no','size'),mp=('p','mean')).sort_values('n',ascending=False).head(15)
g['port_mean_p']=va[va.cur.isin(g.index)].groupby('cur').p.mean(); g['port_n']=va[va.cur.isin(g.index)].groupby('cur').size()
print(g.to_string())
print('pctile of mean p overall', np.percentile(p,[50,90,99]))

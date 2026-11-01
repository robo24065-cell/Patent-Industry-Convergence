import sys,numpy as np,pandas as pd,time
sys.stdout.reconfigure(encoding='utf-8')
from common import *; from evalkit import *
import xgboost as xgb
tr=load_panel(range(2006,2012)); va=load_panel(VAL_L)
feats=['age','age_appl','rgst_year','claims','orig_type','cur_type','cur_n','cur_is_orig','prior_MAIN','prior_CORP','yrs_since_change','own_port','own_out3','own_out3_days','own_in3','own_out_all','own_corp3','own_lapse3','own_lapse1','own_lapse_rate','own_out_rate','ksic_rate3','n_pay','paid_thru','prepaid','days_since_pay','amt_per_yr','multi','lic_any','lic_active','lic_excl','n_trial','n_inv','n_inv_for','n_app','n_prio','for_prio','inv_is_app','exam_yrs','has_agent']
for d in (tr,va): d['ksic_c']=d.ksic.astype(int); d['pct']=(d.applTpcd=='국제출원').astype(int)
F=feats+['ksic_c','pct']
t=time.time()
m=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=20,nthread=2,tree_method='hist',eval_metric='auc')
m.fit(tr[F],tr.y_MAIN); p=m.predict_proba(va[F])[:,1]
print(time.time()-t, {k:round(v,4) for k,v in metrics(va.y_MAIN,p).items()})
imp=pd.Series(m.get_booster().get_score(importance_type='gain')).sort_values(ascending=False); print(imp.head(20).round(1).to_string())
# 번들 여부: 같은 양도인·같은 날 양도 건수
pos=va[va.y_MAIN==1].copy(); pos['bk']=pos.groupby(['L','cur','y_date']).rgst_no.transform('size')
print(pd.cut(pos.bk,[0,1,2,5,20,100,10000]).value_counts().sort_index())
va['bundle']=0; va.loc[pos.index,'bundle']=pos.bk.values
for lo,hi in [(1,1),(2,5),(6,50),(51,99999)]:
    k=(va.y_MAIN==0)|((va.bundle>=lo)&(va.bundle<=hi))
    print('bundle',lo,hi,'pos',int(((va.bundle>=lo)&(va.bundle<=hi)).sum()),'AUC',round(roc_auc_score(va.y_MAIN[k],p[k]),4))
print(seg_report(va,p,'cur_type').to_string()); print(seg_report(va,p,'L').to_string())

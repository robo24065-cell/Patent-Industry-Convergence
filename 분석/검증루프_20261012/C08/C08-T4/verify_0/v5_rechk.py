# C08-T4 verify_0 v5 (2nd refuter): P1 recount from labels, k_late1 rebuild from kr_registered KSIC, Δskill recompute with sklearn AUC + independent cluster bootstrap (resample clusters), 3-seed ensemble.
import sys, os, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import roc_auc_score
T=time.time()
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
CW=B+r'\분석\검증루프_20261012\C08\C08-T4\cache'
OUT=os.path.dirname(os.path.abspath(__file__))
R={}
L=pd.read_parquet(os.path.join(CW,'s1_labels.parquet'))
dev=L[L.origin.between(2015,2022)]
g=dev.groupby('origin').late7.agg(['size','sum','mean'])
R['P1_pos']=g['sum'].astype(int).to_dict(); R['P1_total']=int(g['sum'].sum()); R['P1_min']=int(g['sum'].min())
R['P1_rate_mean']=float(g['mean'].mean()); r2122=float(g.loc[[2021,2022],'mean'].mean()); R['lock_exp']=338978*r2122; R['rate2122']=r2122
R['uniq']=int(dev[dev.late7==1].rgst_no.nunique())
# k_late1 rebuild from provided kr_registered KSIC
kr=pd.read_parquet(os.path.join(B,'공모전데이터','작업데이터','kr_registered.parquet'),columns=['등록번호','세세분류코드'])
kr['등록번호']=kr['등록번호'].astype(str).str.strip(); mk=kr.drop_duplicates('등록번호').set_index('등록번호')['세세분류코드'].astype(str)
R['kr_dup_rgst']=int(kr['등록번호'].duplicated().sum()); R['n_ksic']=int(mk.nunique())
L['k']=L.rgst_no.map(mk)
R['label_ksic_missing']=float(L.k.isna().mean())
rate=L.groupby(['origin','k']).late7.mean()
mx={}
for Y in range(2015,2023):
    F=pd.read_parquet(os.path.join(CW,f'feat_{Y}.parquet'),columns=['rgst_no','y','k_late1_prov','k_trend_prov','k_aband1_prov','ksic'])
    mine=F.rgst_no.map(mk).map(rate.loc[Y-1])
    mx[Y]=float(np.nanmax(np.abs(mine.values-F.k_late1_prov.values)))
    R[f'{Y}_klate1_nuniq']=int(F.k_late1_prov.nunique()); R[f'{Y}_auc_klate1']=float(roc_auc_score(F.y,F.k_late1_prov))
    # y match with labels
    yl=F.rgst_no.map(L[L.origin==Y].set_index('rgst_no').late7)
    R[f'{Y}_y_mismatch']=int((yl.values!=F.y.values).sum())
R['klate1_maxabsdiff']=mx
# Δskill recompute
rng=np.random.default_rng(777)
ens={}
for a in ['20261012','20261013','20261014']:
    P=pd.read_parquet(os.path.join(CW,f'p4_pred_{a}.parquet')).reset_index(drop=True)
    ens[a]=P
    a0=P.groupby('origin').apply(lambda d: roc_auc_score(d.y,d.M0)); a1=P.groupby('origin').apply(lambda d: roc_auc_score(d.y,d.M1))
    R[f'{a}_dskill']=float(((a1-a0)/(1-a0)).mean()); R[f'{a}_aucM0']=float(a0.mean()); R[f'{a}_aucM1']=float(a1.mean())
    R[f'{a}_n_eval']=int(len(P)); R[f'{a}_pos_eval']=int(P.y.sum())
# ensemble (mean of 3 seeds' probabilities) with cluster bootstrap by resampling clusters (different impl)
P=ens['20261012'][['rgst_no','origin','y','cd']].copy()
for m in ['M0','M1']:
    P[m]=np.mean([ens[a][m].values for a in ens],axis=0)
    assert all((ens[a].rgst_no.values==P.rgst_no.values).all() for a in ens)
a0=P.groupby('origin').apply(lambda d: roc_auc_score(d.y,d.M0)); a1=P.groupby('origin').apply(lambda d: roc_auc_score(d.y,d.M1))
R['ens_dskill']=float(((a1-a0)/(1-a0)).mean()); R['ens_by_origin']={int(k):round(float(x),5) for k,x in ((a1-a0)/(1-a0)).items()}
cl=pd.Index(P.cd.unique()); ci=cl.get_indexer(P.cd); nc=len(cl)
bs=[]
from sklearn.metrics import roc_auc_score as AUC
for b in range(100):
    cnt=np.bincount(rng.integers(0,nc,nc),minlength=nc).astype(float); w=cnt[ci]
    s=[]
    for v,d in P.groupby('origin'):
        ww=w[d.index.values]; k=ww>0
        A0=AUC(d.y.values[k],d.M0.values[k],sample_weight=ww[k]); A1=AUC(d.y.values[k],d.M1.values[k],sample_weight=ww[k])
        s.append((A1-A0)/(1-A0))
    bs.append(np.mean(s))
R['ens_ci']=[float(np.quantile(bs,0.025)),float(np.quantile(bs,0.975))]
R['n_clusters']=nc; R['top_cluster_share_eval']=float(P.cd.value_counts(normalize=True).iloc[0]); R['top_cluster']=str(P.cd.value_counts().index[0])
R['sec']=round(time.time()-T,1)
json.dump(R,open(os.path.join(OUT,'v5_out.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
print(json.dumps(R,ensure_ascii=False,indent=1,default=float))

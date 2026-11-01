# 감사: 시점 누수 검정 — t0 이하 자료만으로 만든 캐시(cache_audit2011)의 피처 == 전체(≤2018) 캐시 피처인가
import os, sys, numpy as np, pandas as pd
os.environ.update(T3_CACHE='cache_audit2011',T3_MAXYEAR='2011',T3_LABEL_MAX='2011')
import t3lib as L
full=pd.concat([pd.read_parquet(f'cache/feat_{t}.parquet') for t in (2008,2011)])
f2full=pd.read_parquet('cache/feat2.parquet'); f2full=f2full[f2full.t0.isin([2008,2011])]
P=L.load()
tr=pd.concat([L.feats(P,t,10) for t in (2008,2011)])
f2tr=pd.read_parquet('cache_audit2011/feat2.parquet'); f2tr=f2tr[f2tr.t0.isin([2008,2011])]
def cmp(a,b,keys,name):
    m=a.merge(b,on=keys,suffixes=('_full','_trunc'),how='outer',indicator=True)
    print(name,'행: full',len(a),'trunc',len(b),m._merge.value_counts().to_dict())
    m=m[m._merge=='both']; bad=[]
    for c in a.columns:
        if c in keys or c+'_full' not in m: continue
        x=pd.to_numeric(m[c+'_full'],errors='coerce'); y=pd.to_numeric(m[c+'_trunc'],errors='coerce')
        diff=~(np.isclose(x,y,rtol=1e-5,atol=1e-6,equal_nan=True))
        if diff.any(): bad.append((c,int(diff.sum())))
    print('  불일치 컬럼:',bad if bad else '없음')
cmp(full.drop(columns=[c for c in ['kr_fut','y'] if c in full]),tr,['cell','t0'],'t3lib 피처')
cmp(f2full,f2tr,['ksic','mg','t0'],'feat2 피처')

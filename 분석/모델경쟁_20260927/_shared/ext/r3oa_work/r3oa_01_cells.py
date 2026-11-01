# -*- coding: utf-8 -*-
"""r3oa 01: T3 셀 유니버스·활동량 순위 (출원연도 <=2016 자료만 사용; T3 R2 cache_dev 읽기 전용)"""
import pandas as pd, numpy as np, os
T3=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R2\cache_dev\panel_ccy.parquet'
OUT=os.path.dirname(os.path.abspath(__file__))
P=pd.read_parquet(T3,columns=['ksic','mg','cc','year','n'])
P=P[P.year<=2016]
tot=P.groupby(['ksic','mg','year']).n.sum().unstack(fill_value=0)
rows=[]
for t0 in range(2010,2017):
    s=tot[[y for y in (t0-2,t0-1,t0) if y in tot.columns]].sum(1)
    e=s[s>=50]
    rows.append(pd.DataFrame({'ksic':e.index.get_level_values(0),'mg':e.index.get_level_values(1),'t0':t0,'n5_3y':e.values}))
E=pd.concat(rows)
print('eligible cell-origin pairs',len(E),'unique cells',E[['ksic','mg']].drop_duplicates().shape[0],'unique mg',E.mg.nunique())
kr=P[P.cc=='KR'].groupby(['ksic','mg']).n.sum()
act=P[P.year.between(2012,2016)].groupby(['ksic','mg']).n.sum()
C=E.groupby(['ksic','mg']).agg(n_orig=('t0','size'),n5_max=('n5_3y','max')).reset_index()
C['act5_1216']=C.set_index(['ksic','mg']).index.map(act).fillna(0).values
C['kr_0016']=C.set_index(['ksic','mg']).index.map(kr).fillna(0).values
C.to_csv(os.path.join(OUT,'r3oa_cells_eligible_dev.csv'),index=False)
M=C.groupby('mg').agg(n_cells=('ksic','size'),cell_orig=('n_orig','sum'),act=('act5_1216','sum'),ksics=('ksic',lambda x:'|'.join(sorted(x)))).reset_index()
M=M.sort_values(['cell_orig','act'],ascending=False).reset_index(drop=True)
M['cum_cell_orig_share']=M.cell_orig.cumsum()/M.cell_orig.sum()
M.to_csv(os.path.join(OUT,'r3oa_mg_rank.csv'),index=False)
for k in (100,200,250,300,400,450,600,1000): print(k, round(M.cum_cell_orig_share.iloc[min(k,len(M))-1],3))
print(M.head(20))

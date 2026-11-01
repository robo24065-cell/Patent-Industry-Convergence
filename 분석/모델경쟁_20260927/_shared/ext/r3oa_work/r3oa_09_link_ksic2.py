# -*- coding: utf-8 -*-
"""r3oa 09 (06의 v2, 2026-09-29 재시도: 균형 보강 98개 포함, 변형별 수집 여부 추가): 50 KSIC별 연계율 (개발 원점 2010–2016 적격 셀 기준, 출원연도 <=2016 특허만 사용)
+ 셀↔메인그룹 연결표 r3oa_cell_link.csv(셀 → 메인그룹 검색어·수집 여부·타당도)."""
import pandas as pd, numpy as np, os
W=os.path.dirname(os.path.abspath(__file__)); EXT=os.path.dirname(W)
C=pd.read_csv(os.path.join(W,'r3oa_cells_eligible_dev.csv'),dtype={'ksic':str})
OA=pd.read_parquet(os.path.join(EXT,'r3oa_mg_year_ALL.parquet'))
V=pd.read_csv(os.path.join(W,'r3oa_query_validity_v2.csv'))
q=OA.groupby('mg').agg(query=('query','first'),works_2000_2016=('n_works',lambda s: 0)).reset_index()
tot=OA[OA.year.between(2000,2016)].groupby('mg').n_works.sum()
q['works_2000_2016']=q.mg.map(tot)
C['covered']=C.mg.isin(q.mg)
for v in ('company','KR'):
    f=os.path.join(EXT,'r3oa_mg_year_%s.parquet'%v)
    C['cov_'+v]=C.mg.isin(pd.read_parquet(f).mg) if os.path.exists(f) else False
L=C.merge(q,on='mg',how='left').merge(V[['mg','hit_self','self_rank']],on='mg',how='left')
L.to_csv(os.path.join(EXT,'r3oa_cell_link_v2.csv'),index=False,encoding='utf-8-sig')
K=C.groupby('ksic').apply(lambda d: pd.Series(dict(cells=len(d),cells_cov=int(d.covered.sum()),
     cell_share=d.covered.mean(),act_share=d.act5_1216[d.covered].sum()/max(d.act5_1216.sum(),1),
     kr_share=d.kr_0016[d.covered].sum()/max(d.kr_0016.sum(),1),
     act_share_company=d.act5_1216[d.cov_company].sum()/max(d.act5_1216.sum(),1),
     act_share_KR=d.act5_1216[d.cov_KR].sum()/max(d.act5_1216.sum(),1)))).reset_index()
K.to_csv(os.path.join(EXT,'r3oa_link_ksic50_v2.csv'),index=False)
print(K.describe().T)
print('KSIC with >=1 covered cell',(K.cells_cov>0).sum(),'/',len(K),'; act_share>=50%:',(K.act_share>=0.5).sum())
print(K.sort_values('act_share').head(10).to_string())
print('overall: cells',len(C),'covered',C.covered.sum(),C.covered.mean(),'act share',C.act5_1216[C.covered].sum()/C.act5_1216.sum(),'kr share',C.kr_0016[C.covered].sum()/C.kr_0016.sum())

for v in ('company','KR'):
    print(v,'KSIC>=1 cell',(K['act_share_'+v]>0).sum(),'act>=50%',(K['act_share_'+v]>=0.5).sum(),'overall act share',C.act5_1216[C['cov_'+v]].sum()/C.act5_1216.sum(),'cells',int(C['cov_'+v].sum()))

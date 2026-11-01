# -*- coding: utf-8 -*-
"""r3oa 11: T3 결합용 메인그룹×원점 피처표 (2026-09-29 재시도). 원점 t0의 피처는 발행연도 <= t0 논문만 사용.
t0 = 2005..2021 (T3 R3 최종 원점 t0=2021까지 — 피처 창이 t0에서 끝나므로 결과변수와 무관).
oa_all_g3  : 전 세계 논문 점유 3년 성장 log((Σt0-2..t0+1)/총량) − log((Σt0-5..t0-3+1)/총량)
oa_all_lv3 : log(Σt0-2..t0 +1)
oa_co_g3   : 기업 소속 논문 점유 성장(분모 = science_ 기업 논문 총량 TOTAL company_ALL)
oa_co_share: log(기업 Σt0-2..t0 +1) − log(전체 Σt0-2..t0 +1)  (기업 비중 수준)
미수집 메인그룹은 행 없음(결합 시 NaN). OpenAlex 2026-09 스냅샷 = 준-정합."""
import pandas as pd, numpy as np, os
W=os.path.dirname(os.path.abspath(__file__)); EXT=os.path.dirname(W)
A=pd.read_parquet(os.path.join(EXT,'r3oa_mg_year_ALL.parquet')).pivot_table(index='mg',columns='year',values='n_works',aggfunc='first')
C=pd.read_parquet(os.path.join(EXT,'r3oa_mg_year_company.parquet')).pivot_table(index='mg',columns='year',values='n_works',aggfunc='first')
T=pd.read_parquet(os.path.join(EXT,'science_openalex_country_total_year.parquet')); ta=T[T.country=='ALL'].set_index('year').n_works_total.sort_index()
E=pd.read_parquet(os.path.join(EXT,'science_openalex_ksic_extra_year.parquet')); tc=E[(E.ksic=='TOTAL')&(E.variant=='company_ALL')].set_index('year').n_works.sort_index()
def win(P,t0,lo,hi): return P[[y for y in range(t0+lo,t0+hi+1)]].sum(1)
rows=[]
for t0 in range(2005,2022):
    a3=win(A,t0,-2,0); a0=win(A,t0,-5,-3)
    d=pd.DataFrame({'mg':A.index,'t0':t0,
        'oa_all_g3':(np.log((a3+1)/ta.loc[t0-2:t0].sum())-np.log((a0+1)/ta.loc[t0-5:t0-3].sum())).values,
        'oa_all_lv3':np.log(a3+1).values})
    c3=win(C,t0,-2,0); c0=win(C,t0,-5,-3)
    dc=pd.DataFrame({'mg':C.index,'oa_co_g3':(np.log((c3+1)/tc.loc[t0-2:t0].sum())-np.log((c0+1)/tc.loc[t0-5:t0-3].sum())).values,
        'oa_co_share':(np.log(c3+1)-np.log(a3.reindex(C.index)+1)).values})
    rows.append(d.merge(dc,on='mg',how='left'))
F=pd.concat(rows,ignore_index=True)
F.to_parquet(os.path.join(EXT,'r3oa_mg_t0_features.parquet'),index=False)
print(F.shape, F.mg.nunique(), F.isna().mean().round(3).to_dict()); print(F.describe().T.round(3))

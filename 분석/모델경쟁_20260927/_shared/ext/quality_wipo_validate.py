# 제공데이터 공개연도 건수 vs WIPO 관청별 공개건수(기술분야별) — 커버리지 비율·추세 일치도
import pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr
d=pd.read_parquet('quality_provided_cc_year_wipo35.parquet')
w=pd.read_csv('quality_raw/quality_wipo_pi/dc_indicator_patent_4_publication_by_technology.csv')
w=w[w.office.isin(['KR','US','JP','CN','EP'])&(w.tec_id>0)].groupby(['office','year','tec_id'])['count'].sum().reset_index()
w.columns=['cc','year','wipo_field','wipo_pub']
pp=d[d.wipo_field>0].groupby(['cc','pub_year','wipo_field']).n.sum().reset_index().rename(columns={'pub_year':'year','n':'prov_pub'})
m=w.merge(pp,on=['cc','year','wipo_field'],how='left').fillna({'prov_pub':0})
m['ratio']=m.prov_pub/m.wipo_pub
m.to_parquet('quality_wipo_vs_provided_pub.parquet',index=False)
tot=m.groupby(['cc','year'])[['wipo_pub','prov_pub']].sum(); tot['ratio']=tot.prov_pub/tot.wipo_pub
tot.reset_index().to_csv('quality_wipo_vs_provided_pub_total.csv',index=False)
# 분야별 3년 성장률 일치도 (2006-2016)
out=[]
for cc in ['KR','US','JP','EP','CN']:
    x=m[(m.cc==cc)].pivot_table(index='wipo_field',columns='year',values=['wipo_pub','prov_pub'])
    rs=[]
    for y in range(2008 if cc!='CN' else 2014,2017):
        a=np.log((x['prov_pub'][y]+1)/(x['prov_pub'][y-3]+1)); b=np.log(x['wipo_pub'][y]/x['wipo_pub'][y-3])
        big=x['prov_pub'][y-3]>=200
        rs.append(spearmanr(a[big],b[big]).statistic)
    out.append((cc, round(np.nanmean(rs),3), round(np.nanmin(rs),3)))
print('분야별 3년 성장률 순위상관(제공 vs WIPO), 평균/최소:', out)
print(tot.ratio.unstack(0).round(3).loc[2014:].to_string())

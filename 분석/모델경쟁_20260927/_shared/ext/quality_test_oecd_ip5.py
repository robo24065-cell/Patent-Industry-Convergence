# 의미 검정: OECD IP5 패밀리(국제 고품질 출원)의 분야별 성장·한국 점유율이 이후 제공데이터 KR 출원 성장을 설명하나 (WIPO 35분야, 개발 <=2016)
import pandas as pd, numpy as np, statsmodels.formula.api as smf
o=pd.read_parquet('quality_oecd_patents_wipo35.parquet')
o=o[(o.PATENT_AUTHORITIES=='9P50_3')&(o.AGENT_ROLE=='INVENTOR')&(o.WIPO!='_T')]
o['f']=o.WIPO.str[-2:].astype(int)
ip=o.pivot_table(index=['f','TIME_PERIOD'],columns='REF_AREA',values='OBS_VALUE').reset_index()
p=pd.read_parquet('quality_provided_cc_year_wipo35.parquet'); p=p[p.wipo_field>0]
pv=p.groupby(['wipo_field','appl_year','cc']).n.sum().unstack('cc').fillna(0).reset_index().rename(columns={'wipo_field':'f','appl_year':'y'})
def S(df,f,col,a,b,ycol): x=df[(df.f==f)&(df[ycol]>=a)&(df[ycol]<=b)][col].sum(); return x
rows=[]
for t0 in [2006,2008,2010,2012]:
    L=3  # IP5 우선권연도 공표 시차 가정: t0 시점에 t0-3까지 이용
    for f in range(1,36):
        w1=S(ip,f,'W',t0-L-2,t0-L,'TIME_PERIOD'); w0=S(ip,f,'W',t0-L-5,t0-L-3,'TIME_PERIOD')
        k1=S(ip,f,'KOR',t0-L-2,t0-L,'TIME_PERIOD')
        kr_post=S(pv,f,'KR',t0+1,t0+3,'y'); kr_pre=S(pv,f,'KR',t0-2,t0,'y'); kr_pre2=S(pv,f,'KR',t0-5,t0-3,'y')
        fo=lambda a,b: sum(S(pv,f,c,a,b,'y') for c in ['US','EP','JP'])
        rows.append(dict(t0=t0,f=f,g_ip5=np.log(w1/w0),kr_ip5_share=k1/w1,kr_post=kr_post,kr_pre=kr_pre,
            y=np.log1p(kr_post)-np.log1p(kr_pre),g_kr=np.log1p(kr_pre)-np.log1p(kr_pre2),g_f=np.log1p(fo(t0-2,t0))-np.log1p(fo(t0-5,t0-3)),lkr=np.log1p(kr_pre)))
d=pd.DataFrame(rows); d=d[d.kr_pre>=30]
d.to_csv('quality_oecd_ip5_field_test.csv',index=False)
base='y ~ g_kr + g_f + lkr + C(t0)'
for add in ['','+ g_ip5','+ kr_ip5_share','+ g_ip5 + kr_ip5_share']:
    m=smf.ols(base+add,d).fit(cov_type='cluster',cov_kwds={'groups':d.f})
    print(add or 'base', 'n',len(d),'R2',round(m.rsquared,4), {k:(round(m.params[k],3),round(m.tvalues[k],2)) for k in ['g_ip5','kr_ip5_share','g_f'] if k in m.params})
for t0,g in d.groupby('t0'):
    m=smf.ols('y ~ g_kr + g_f + lkr + g_ip5 + kr_ip5_share',g).fit()
    print(t0, {k:(round(m.params[k],3),round(m.tvalues[k],2)) for k in ['g_ip5','kr_ip5_share']})

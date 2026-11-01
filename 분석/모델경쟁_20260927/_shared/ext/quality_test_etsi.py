# 의미 검정: 기술셀의 SEP 선언 비중(원점까지 선언된 것만)이 이후 KR 출원 성장을 설명하나 — SEP 연계 KSIC(>=100건) 한정
import duckdb, pandas as pd, numpy as np, statsmodels.formula.api as smf
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
ks=pd.read_csv('quality_etsi_sep_link_by_ksic.csv',dtype={'ksic':str}); ks=ks[ks.n_sep>=100].ksic.tolist()
con.sql(f"""create table p as select distinct cc, appl_no, appl_year, ksic, trim(split_part(main_ipc,'/',1)) mg from read_parquet('../meta/*.parquet')
 where main_ipc is not null and appl_year between 1995 and 2016 and ksic in ({",".join("'"+k+"'" for k in ks)})""")
con.sql("create table s as select distinct cc, appl_no, ksic, first_decl from 'quality_etsi_sep_link.parquet'")
res=[]
for t0 in [2006,2008,2010,2012]:
    d=con.sql(f"""select p.ksic, p.mg,
      count(distinct case when appl_year between {t0-5} and {t0} then p.cc||p.appl_no end) n_all,
      count(distinct case when appl_year between {t0-5} and {t0} and s.first_decl<=DATE '{t0}-12-31' then p.cc||p.appl_no end) n_sep,
      count(distinct case when p.cc='KR' and appl_year between {t0-2} and {t0} then p.appl_no end) kr_pre,
      count(distinct case when p.cc='KR' and appl_year between {t0-5} and {t0-3} then p.appl_no end) kr_pre2,
      count(distinct case when p.cc in('US','EP','JP') and appl_year between {t0-2} and {t0} then p.cc||p.appl_no end) f_pre,
      count(distinct case when p.cc in('US','EP','JP') and appl_year between {t0-5} and {t0-3} then p.cc||p.appl_no end) f_pre2,
      count(distinct case when p.cc='KR' and appl_year between {t0+1} and {t0+3} then p.appl_no end) kr_post
     from p left join s on p.cc=s.cc and p.appl_no=s.appl_no and p.ksic=s.ksic group by 1,2 having n_all>=30""").df()
    d['t0']=t0; res.append(d)
d=pd.concat(res)
d['y']=np.log1p(d.kr_post)-np.log1p(d.kr_pre); d['lsize']=np.log1p(d.n_all); d['lkr']=np.log1p(d.kr_pre)
d['g_kr']=np.log1p(d.kr_pre)-np.log1p(d.kr_pre2); d['g_f']=np.log1p(d.f_pre)-np.log1p(d.f_pre2); d['sep_share']=d.n_sep/d.n_all; d['has_sep']=(d.n_sep>0).astype(int)
d.to_parquet('quality_etsi_cell_test.parquet',index=False)
g=d.ksic.astype('category').cat.codes
base='y ~ lsize + lkr + g_kr + g_f + C(t0) + C(ksic)'
m0=smf.ols(base,d).fit(cov_type='cluster',cov_kwds={'groups':g})
for add in ['sep_share','has_sep']:
    m=smf.ols(base+' + '+add,d).fit(cov_type='cluster',cov_kwds={'groups':g})
    print(add,'n',len(d),'cells with SEP',int(d.has_sep.sum()),'R2',round(m0.rsquared,4),'->',round(m.rsquared,4),'coef',round(m.params[add],3),'t',round(m.tvalues[add],2))
for t0,gg in d.groupby('t0'):
    m=smf.ols('y ~ lsize + lkr + g_kr + g_f + C(ksic) + sep_share',gg).fit(); print(t0,len(gg),int(gg.has_sep.sum()),round(m.params['sep_share'],3),round(m.tvalues['sep_share'],2))

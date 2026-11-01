# 의미 검정: 기술셀(KSIC x IPC 메인그룹)의 '시장가치(KPSS xi) 집약도'가 이후 KR 출원 성장을 설명하나 (개발 구간 <=2016)
import duckdb, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
con.sql("""create table p as select distinct cc, appl_no, appl_year, ksic, trim(split_part(main_ipc,'/',1)) mg
 from read_parquet('../meta/*.parquet') where main_ipc is not null and appl_year between 1998 and 2016""")
con.sql("""create table l as select distinct cc, appl_no, ksic, us_pn, year(issue_date) iy, xi_real, cites from 'quality_kpss_family_link.parquet'""")
con.sql("create table lm as select l.*, trim(split_part(p.mg,'/',1)) mg from l join (select distinct cc, appl_no, ksic, mg from p) p using(cc,appl_no,ksic)")
res=[]
for t0 in [2006,2008,2010,2012]:
    d=con.sql(f"""
    with base as (select ksic, mg,
        count(distinct case when appl_year between {t0-3} and {t0} then cc||appl_no end) n_all,
        count(distinct case when cc='KR' and appl_year between {t0-3} and {t0} then appl_no end) kr_pre,
        count(distinct case when cc='KR' and appl_year between {t0-7} and {t0-4} then appl_no end) kr_pre2,
        count(distinct case when cc in('US','EP','JP') and appl_year between {t0-3} and {t0} then cc||appl_no end) f_pre,
        count(distinct case when cc in('US','EP','JP') and appl_year between {t0-7} and {t0-4} then cc||appl_no end) f_pre2,
        count(distinct case when cc='KR' and appl_year between {t0+1} and {t0+3} then appl_no end) kr_post
      from p group by 1,2),
    v as (select ksic, mg, count(distinct us_pn) n_val, avg(ln(xi_real+0.01)) lxi, median(xi_real) mxi, avg(ln(1+cites)) lcit
      from lm where iy between {t0-4} and {t0} group by 1,2)
    select base.*, coalesce(n_val,0) n_val, lxi, mxi, lcit from base left join v using(ksic,mg) where n_all>=20""").df()
    d['t0']=t0; res.append(d)
d=pd.concat(res)
d['y']=np.log1p(d.kr_post)-np.log1p(d.kr_pre)
d['lsize']=np.log1p(d.n_all); d['lkr']=np.log1p(d.kr_pre); d['g_kr']=np.log1p(d.kr_pre)-np.log1p(d.kr_pre2); d['g_f']=np.log1p(d.f_pre)-np.log1p(d.f_pre2)
d['val_share']=d.n_val/d.n_all
d['has_val']=(d.n_val>=3).astype(int)
d['lxi_f']=d.lxi.where(d.n_val>=3, np.nan)
d.to_parquet('quality_kpss_cell_test.parquet',index=False)
print('cells', d.groupby('t0').size().to_dict(), ' with >=3 valued', d.groupby('t0').has_val.sum().to_dict())
out=[]
base='y ~ lsize + lkr + g_kr + g_f + C(t0)'
m0=smf.ols(base,d).fit(cov_type='cluster',cov_kwds={'groups':d.ksic.astype('category').cat.codes}); out.append(('base',m0.rsquared,None,None))
m1=smf.ols(base+' + val_share',d).fit(cov_type='cluster',cov_kwds={'groups':d.ksic.astype('category').cat.codes}); out.append(('+val_share',m1.rsquared,m1.params['val_share'],m1.tvalues['val_share']))
dd=d[d.n_val>=3].copy(); g=dd.ksic.astype('category').cat.codes
m2a=smf.ols(base,dd).fit(cov_type='cluster',cov_kwds={'groups':g})
m2=smf.ols(base+' + lxi_f',dd).fit(cov_type='cluster',cov_kwds={'groups':g}); out.append(('valued cells base',m2a.rsquared,None,None)); out.append(('+lxi (valued cells)',m2.rsquared,m2.params['lxi_f'],m2.tvalues['lxi_f']))
m3=smf.ols(base+' + lcit',dd).fit(cov_type='cluster',cov_kwds={'groups':g}); out.append(('+lcit (non-PIT, 해석용)',m3.rsquared,m3.params['lcit'],m3.tvalues['lcit']))
for r in out: print(r)
# 원점별 부호 일관성
for t0,gg in dd.groupby('t0'):
    mm=smf.ols('y ~ lsize + lkr + g_kr + g_f + lxi_f',gg).fit(); rs=spearmanr(gg.lxi_f, mm.resid+mm.params['lxi_f']*gg.lxi_f)
    ms=smf.ols('y ~ lsize + lkr + g_kr + g_f + val_share',d[d.t0==t0]).fit()
    print(t0, 'n',len(gg),'coef lxi',round(mm.params['lxi_f'],4),'t',round(mm.tvalues['lxi_f'],2), '| val_share coef',round(ms.params['val_share'],3),'t',round(ms.tvalues['val_share'],2))

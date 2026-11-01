import pandas as pd, numpy as np
C='cache/'
ap=pd.read_parquet(C+'applicant.parquet'); pt=pd.read_parquet(C+'patent.parquet'); pay=pd.read_parquet(C+'pay.parquet'); rt=pd.read_parquet(C+'right.parquet')
a1=ap[ap.seq==0].copy()
a1['d1']=a1.applicantCd.str[0]
#print(a1.d1.value_counts().head(10))
for d in []:# ['1','2','3','4','5','6','7','8','9']:
    pass#[a1.d1==d]
    pass#: print(d,len(s),s.applicantNatl.value_counts().head(3).to_dict(), s.applicantName.sample(min(6,len(s)),random_state=0).tolist())
print(ap.groupby('rgst_no').size().value_counts().head())
# pay patterns
#
pay['s']=pd.to_numeric(pay.statAnnl,errors='coerce'); pay['l']=pd.to_numeric(pay.lastAnnl,errors='coerce')
print('span dist', (pay.l-pay.s+1).value_counts().head(10))
print('first seq stat', pay[pay.seq==0].s.value_counts().head(), pay[pay.seq==0].l.value_counts().head())
# lapse lag: E = rgstDate + final lastAnnl years; lag to rgstCsDate. only registers <=2018
lap=rt[(rt.rgstCsName=='소멸등록')&(rt.rgstCsReason.isin(['등록료불납','연차료불납']))&(rt.rgstCsDate<='20181231')]
print('lapse regs <=2018:',len(lap), lap.rgst_no.duplicated().sum())
mx=pay.groupby('rgst_no').l.max().rename('lmax')
x=lap.merge(pt[['rgst_no','rgstDate']],on='rgst_no').merge(mx,on='rgst_no',how='left')
x['rd']=pd.to_datetime(x.rgstDate.str.strip(),format="%Y%m%d",errors="coerce"); x['cd']=pd.to_datetime(x.rgstCsDate.str.strip(),format="%Y%m%d",errors="coerce")
x['E']=[r+pd.DateOffset(years=int(l)) if l==l else pd.NaT for r,l in zip(x.rd,x.lmax)]
x['lag']=(x.cd-x.E).dt.days
print(x.lag.describe(percentiles=[.01,.05,.5,.95,.99]))
print('lmax dist at lapse', x.lmax.value_counts().sort_index().head(25))
print('neg lag examples', x[x.lag<0].head())
print('bad dates', x.cd.isna().sum(), x[x.cd.isna()].rgstCsDate.head().tolist())
# timing of last payment vs E (pay before due)
pay['pd']=pd.to_datetime(pay.payDate.str.strip(),format="%Y%m%d",errors="coerce")
pp=pay.merge(pt[['rgst_no','rgstDate']],on='rgst_no')
pp['rd']=pd.to_datetime(pp.rgstDate.str.strip(),format="%Y%m%d",errors="coerce")
pp=pp[(pp.seq>0)&(pp.pd<='2018-12-31')]
due=[r+pd.DateOffset(years=int(s)-1) for r,s in zip(pp.rd,pp.s)]
pp['due']=due
pp['early']=(pp.due-pp.pd).dt.days
print('days paid before due (seq>0, <=2018):', pp.early.describe(percentiles=[.01,.05,.1,.5,.9,.95,.99]))
print('late pay share (after due)', (pp.early<0).mean(), ' late>0 & <=183', ((pp.early<0)&(pp.early>=-190)).mean())

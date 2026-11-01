# 3단계 사전점검: has_owner(권리자 이력 존재) 판정이 L 이후 정보에 의존하는지 확인 (결과변수 계산 없음)
import sqlite3, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
o=pd.read_sql("select rgst_no,min(ownerRgstCsDate) d0 from owner where ownerRgstCsDate<'20250823' group by rgst_no",con)
p=pd.read_sql("select rgst_no,rgstDate from patent",con)
m=p.merge(o,on='rgst_no',how='left')
m['r']=pd.to_datetime(m.rgstDate,format='%Y%m%d',errors='coerce'); m['d0']=pd.to_datetime(m.d0,format='%Y%m%d',errors='coerce')
print('patents',len(m),'no owner',m.d0.isna().sum())
x=m.dropna(subset=['d0'])
print('first owner row == rgstDate',(x.d0==x.r).mean(),' > rgstDate',(x.d0>x.r).sum(),' < rgstDate',(x.d0<x.r).sum())
for Y in [2006,2011,2014,2016,2019,2020,2021]:
    L=pd.Timestamp(Y,1,1); a=m[m.r<L]
    print(Y,'reg<L',len(a),'has_owner',int(a.d0.notna().sum()),'first_owner>=L (미래의존 포함)',int((a.d0>=L).sum()))

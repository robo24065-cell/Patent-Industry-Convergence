import sys,io,pandas as pd
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
p=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext\r3nts_applicant_status.parquet"
df=pd.read_parquet(p)
print(df.shape, list(df.columns)[:30])
print(df.dtypes.head(30))
df['bz']=df['bizno'].astype(str).str.replace('-','',regex=False)
print(df.bz.str.len().value_counts().head())
corp=df[df.bz.str[3:5].isin(['81','85','86','87','88'])]
ed=pd.to_datetime(corp['end_dt'],errors='coerce')
reg=corp['registered'].fillna(False).astype(bool) if 'registered' in corp else True
off={2016:69600,2017:70362,2018:69667,2019:69587,2020:67547,2021:65983,2022:67656,2023:75668,2024:82931,2025:85270}
c2=corp.assign(ey=ed.dt.year)
for flt,name in ((reg,'registered'),(pd.Series(True,index=corp.index),'all')):
    x=c2[flt]
    for y in (2016,2020,2024,2025):
        n=x[x.ey==y].bz.nunique(); print(name,y,n,round(n/off[y],4))
# also bt 85 branch share among closures
x=c2[reg]
for y in (2016,2025):
    s=x[x.ey==y].drop_duplicates('bz'); print(y,'85 branch share',round((s.bz.str[3:5]=='85').mean(),3))

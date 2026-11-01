import pandas as pd,glob,os
os.chdir(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T1")
d=pd.read_parquet('cache/panels/off_2012.parquet'); print(d.shape, d.columns.tolist()); print(d.head(2).T)
for t in [2012,2016,2020,2022]:
    d=pd.read_parquet(f'cache/panels/off_{t}.parquet'); print(t,len(d), d.y.sum() if 'y' in d else None)
s=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
for f in ['trade_etsi_sep_link.parquet','trade_etsi_sep_patents.parquet']:
  d=pd.read_parquet(os.path.join(s,f)); print(f,d.shape,d.dtypes.to_dict()); print(d.head(2).T)

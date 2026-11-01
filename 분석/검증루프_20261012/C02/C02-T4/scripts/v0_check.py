# v0: 재개 시 중간 파일 점검(라벨 범위·중복 사건)
import sys; sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from lib_cd import *
pat=pd.read_parquet(C+r'\pat.parquet'); print('pat rows',len(pat),'ksic in50',pat.ksic.isin(KSIC50).mean(), pat.ksic.isna().sum(), pat.ksic.nunique())
docs=pd.read_parquet(C+r'\docs.parquet'); print('docs ksic in50',docs.ksic.isin(KSIC50).mean(), docs.ksic.value_counts().head(3).to_dict())
P=pd.read_parquet(C+r'\dev_pos.parquet'); print(P.columns.tolist()); print('pos',len(P),'k in50',P.k.isin(KSIC50).mean())
print('unique (cd,k,rgst_no,d)',P.drop_duplicates(['cd','k','rgst_no','d']).shape[0],' unique (cd,k)',P.drop_duplicates(['cd','k']).shape[0])
print(P.groupby('t0').size().to_dict(), P.groupby('t0').size().sum())
own=pd.read_parquet(C+r'\own_lt2025.parquet'); print('own max d',own.d.max(),'rows',len(own),'acc',own.acc.sum())
print('owner seq0 cs', own[own.seq==0].cs.value_counts(dropna=False).head(3).to_dict())

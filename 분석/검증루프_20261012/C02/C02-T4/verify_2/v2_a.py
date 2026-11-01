import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
C=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\cache'
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
from lib_cd import KSIC50
pat=pd.read_parquet(C+r'\pat.parquet',columns=['rgst_no','ksic','appl_no'])
docs=pd.read_parquet(C+r'\docs.parquet')
lab=pd.read_parquet(C+r'\kr_publab_src.parquet')
print('lab cols',lab.columns.tolist(), len(lab))
print('k_free nunique',lab.k_free.nunique(),'in KSIC50 share',lab.k_free.isin(KSIC50).mean(),'null',lab.k_free.isna().mean())
print('k_free values not in KSIC50', lab.k_free[~lab.k_free.isin(KSIC50)].value_counts().head())
print('pat.ksic in KSIC50', pat.ksic.isin(KSIC50).mean(), pat.ksic.nunique(), pat.ksic.isna().mean())
print('docs.ksic in KSIC50', docs.ksic.isin(KSIC50).mean(), 'n',len(docs))
d1=docs[docs.cd.str[0]=='1']
print('docs prefix1 n',len(d1),'prov lab coverage',d1.ksic.isin(KSIC50).mean(),'public k_free coverage', d1.appl_no.map(lab.drop_duplicates('appl_no').set_index('appl_no').k_free).notna().mean())
print('docs prefix1 by reg', d1.groupby('reg').apply(lambda g: pd.Series({'n':len(g),'prov':g.ksic.isin(KSIC50).mean(),'pub':g.appl_no.isin(set(lab.appl_no[lab.k_free.notna()])).mean()})))
print('lab dup appl_no', lab.appl_no.duplicated().sum())
print('pat appl in lab', pat.appl_no.isin(set(lab.appl_no)).mean())
m=pat.appl_no.map(lab.drop_duplicates('appl_no').set_index('appl_no').k_free)
print('pat agree k_free', (m==pat.ksic).mean())

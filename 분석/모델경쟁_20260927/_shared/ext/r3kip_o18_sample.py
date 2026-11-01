# r3kip_ 재시도 확장: T4 개발 원점 O18(L=2018-01-01, 3년 결과 ≤2020-12) 사례-대조 표본 250건 (O16 결과 재현 검정용)
# 원본은 읽기 전용. 이미 수집한 550건은 제외. 결과: r3kip_o18_sample.parquet
import pandas as pd, numpy as np, os
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
T4C=B+r'\T4_특허포기예측\cache'; T4R2=B+r'\T4_특허포기예측\R2\cache'; OUT=B+r'\_shared\ext'
rng=np.random.default_rng(20260930)
pt=pd.read_parquet(T4C+r'\patent.parquet',columns=['rgst_no','ksic','applNo','rgstDate'])
pt=pt[pt.rgstDate.notna()&(pt.rgstDate!='')&(pt.rgst_no.str[:2]=='10')].reset_index(drop=True)
pt['pid']=np.arange(len(pt))
f=pd.read_parquet(T4R2+r'\feat_20180101.parquet',columns=['pid','ksic','rgst_year','claims','age'])
f['y']=np.load(T4R2+r'\lab_20180101_full.npy')
f=f.merge(pt[['pid','rgst_no','applNo','ksic']].rename(columns={'ksic':'ksic_pt'}),on='pid',how='left')
km=(f.ksic.astype(str)==f.ksic_pt.astype(str)).mean()
print('O18 위험집합',len(f),'KSIC 일치율',round(km,4),'3년 포기율',round(f.y.mean(),4))
assert km>0.999
old=set(pd.read_parquet(OUT+r'\r3kip_sample.parquet').applNo)
f=f[~f.applNo.isin(old)]
extra=set(rng.choice(sorted(f.ksic.astype(str).unique()),25,replace=False))
rows=[]
for k,g in f.groupby('ksic'):
    nk=3 if str(k) in extra else 2
    for lab in (1,0):
        gg=g[g.y==lab]
        rows.append(gg.sample(n=min(nk,len(gg)),random_state=int(rng.integers(1e9))))
s=pd.concat(rows); s['stratum_n']=s.groupby(['ksic','y']).pid.transform('size')
pop=f.groupby(['ksic','y']).size().rename('pop_n').reset_index()
s=s.merge(pop,on=['ksic','y']); s['w']=s.pop_n/s.stratum_n; s['set']='T4_O18'; s['ksic']=s.ksic.astype(str)
s[['applNo','rgst_no','ksic','pid','y','w','set','rgst_year','claims','age']].to_parquet(OUT+r'\r3kip_o18_sample.parquet',index=False)
print('표본',len(s),'포기',int(s.y.sum()),'KSIC',s.ksic.nunique(),'고유 출원',s.applNo.nunique())

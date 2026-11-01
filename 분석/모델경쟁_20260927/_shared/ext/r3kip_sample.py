# r3kip_ 표본 설계: S_T4(O16 사례-대조 400) + S_BW(그 부분집합 150, 후방인용) + S_T3(t0=2016 적격셀 150, 셀당 1건)
# 원본은 읽기 전용. 결과: r3kip_sample.parquet
import pandas as pd, numpy as np, duckdb, os
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927'
T4C=B+r'\T4_특허포기예측\cache'; T4R2=B+r'\T4_특허포기예측\R2\cache'
OUT=B+r'\_shared\ext'
rng=np.random.default_rng(20260929)
pt=pd.read_parquet(T4C+r'\patent.parquet',columns=['rgst_no','ksic','applNo','applDate','rgstDate','ipcCd','claimCount'])
pt=pt[pt.rgstDate.notna()&(pt.rgstDate!='')&(pt.rgst_no.str[:2]=='10')].reset_index(drop=True)
pt['pid']=np.arange(len(pt))
f=pd.read_parquet(T4R2+r'\feat_20160101.parquet',columns=['pid','ksic','rgst_year','claims','age'])
y=np.load(T4R2+r'\lab_20160101_full.npy')
f['y']=y
f=f.merge(pt[['pid','rgst_no','applNo','ksic']].rename(columns={'ksic':'ksic_pt'}),on='pid',how='left')
print('feat',len(f),'ksic 일치율',(f.ksic.astype(str)==f.ksic_pt.astype(str)).mean(),'pos',f.y.mean().round(4))
assert (f.ksic.astype(str)==f.ksic_pt.astype(str)).mean()>0.999
rows=[]
for k,g in f.groupby('ksic'):
    for lab in (1,0):
        gg=g[g.y==lab]
        take=gg.sample(n=min(4,len(gg)),random_state=int(rng.integers(1e9)))
        rows.append(take)
s4=pd.concat(rows); s4['stratum_n']=s4.groupby(['ksic','y']).pid.transform('size')
pop=f.groupby(['ksic','y']).size().rename('pop_n').reset_index()
s4=s4.merge(pop,on=['ksic','y']); s4['w']=s4.pop_n/s4.stratum_n
s4['set']='T4'
# 후방인용 부분집합: KSIC당 3건
bw=s4.groupby('ksic',group_keys=False).apply(lambda g:g.sample(n=min(3,len(g)),random_state=int(rng.integers(1e9))))
s4['bw']=s4.pid.isin(bw.pid)
# T3: t0=2016 개발 적격셀
d=pd.read_parquet(B+r'\T3_기술셀부상예측\R2\preds\dev_preds.parquet')
d=d[d.t0==2016][['cell','ksic','mg','y','yb']]
m=duckdb.sql(f"""select appl_no, ksic, main_ipc, appl_year, rgst_date from read_parquet('{B}/_shared/meta/*.parquet')
 where cc='KR' and appl_year between 2009 and 2012 and rgst_no is not null and substr(appl_no,1,2)='10'""").df()
m['mg']=m.main_ipc.str.split('/').str[0].str.strip()
m['cell']=m.ksic.astype(str)+'|'+m.mg
m=m[m.cell.isin(d.cell)&~m.appl_no.isin(s4.applNo)]
cand=d[d.cell.isin(m.cell)]
print('t0=2016 적격셀',len(d),'KR 2009-12 등록특허 보유 셀',len(cand))
cs=cand.sample(n=150,random_state=int(rng.integers(1e9)))
t3=m[m.cell.isin(cs.cell)].groupby('cell',group_keys=False).apply(lambda g:g.sample(n=1,random_state=int(rng.integers(1e9))))
t3=t3.merge(cs[['cell','y','yb']],on='cell').rename(columns={'appl_no':'applNo','y':'t3_y','yb':'t3_yb'})
t3['set']='T3'
S=pd.concat([s4[['applNo','rgst_no','ksic','pid','y','w','set','bw','rgst_year','claims','age']],
             t3[['applNo','ksic','cell','mg','t3_y','t3_yb','set']]],ignore_index=True)
S['bw']=S.bw.fillna(False).astype(bool)
S['ksic']=S.ksic.astype(str)
S.to_parquet(OUT+r'\r3kip_sample.parquet',index=False)
print(S.groupby('set').size(), 'bw',S.bw.sum(), 'T4 pos',S[S.set=='T4'].y.mean(), 'uniq appl',S.applNo.nunique())

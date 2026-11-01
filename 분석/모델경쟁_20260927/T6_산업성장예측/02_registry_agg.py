# registry_full(읽기전용) → (ksic, year) KR 등록·소멸(불납)·이전 사건 수와 연초 존속 스톡. 사건일 기준(그 시점에 공개된 등록원부 사건) → 누수 아님.
import sqlite3, pandas as pd, numpy as np, time
t=time.time()
p=r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/수집파이프라인/registry_full.sqlite"
c=sqlite3.connect('file:'+p+'?mode=ro',uri=True)
pat=pd.read_sql('select rgst_no, ksic, substr(rgstDate,1,4) ry, substr(applDate,1,4) ay from patent',c)
print(pat.ry.value_counts().sort_index().to_string())
lap=pd.read_sql("select rgst_no, substr(rgstCsDate,1,4) y from right where rgstCsName='소멸등록' and rgstCsReason='등록료불납'",c)
tr=pd.read_sql("select rgst_no, substr(ownerRgstCsDate,1,4) y from owner where ownerRgstCsReason like '%이전%'",c)
# 한 특허가 여러 KSIC에 걸칠 수 있음 → patent 테이블 행 단위로 ksic 부여
pat['ksic']=pat.ksic.astype(str)
def agg(ev,name):
    m=ev.merge(pat[['rgst_no','ksic']],on='rgst_no')
    return m.groupby(['ksic','y']).size().rename(name)
reg=pat.groupby(['ksic','ry']).size().rename('kr_new_rgst'); reg.index.names=['ksic','y']
out=pd.concat([reg,agg(lap,'kr_lapse'),agg(tr,'kr_transfer')],axis=1).fillna(0).reset_index()
out['y']=pd.to_numeric(out.y,errors='coerce'); out=out.dropna(subset=['y']); out['y']=out.y.astype(int)
out=out[(out.y>=2000)&(out.y<=2024)].rename(columns={'y':'year'})
out.to_parquet('cache/kr_registry_ksic_year.parquet',index=False)
print(out.groupby('year')[['kr_new_rgst','kr_lapse','kr_transfer']].sum().to_string())
print('sec',time.time()-t)

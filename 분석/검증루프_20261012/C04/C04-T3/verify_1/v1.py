# C04-T3 verify_1: 개발 원점 SAB 추세 독립 재계산(반박 검증). s1/s2/s3 코드·캐시 미사용. meta 직접 + 동결 예측.
import os, sys, json, hashlib, html, unicodedata, time
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8'); t=time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
V = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3\verify_1")
META = os.path.join(ROOT, r"모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
PR = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\preds")
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
q = f"""
with r as (
  select cc, ksic, appl_no, applicant,
     regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', '', 'g') ip,
     least(nullif(pub_date,''), nullif(rgst_date,'')) d1, nullif(pub_date,'') d2, nullif(rgst_date,'') d3
  from read_parquet('{META}') where appl_year >= 2000),
s as (select cc, ksic, appl_no, applicant, ip, try_cast(left(coalesce(d1,d2,d3),4) as int) yr from r),
t as (select cc, ksic, appl_no, applicant, yr,
     case when regexp_matches(ip, '^[A-H][0-9]{{2}}[A-Z][0-9]{{1,4}}/') then substr(ip,1,4)||' '||cast(cast(regexp_extract(ip,'^[A-H][0-9]{{2}}[A-Z]([0-9]{{1,4}})/',1) as int) as varchar)
          when regexp_matches(ip, '^[A-H][0-9]{{2}}[A-Z][0-9]{{3,6}}$') then substr(ip,1,4)||' '||cast(cast(substr(ip,5,length(ip)-6) as int) as varchar) end mg
  from s where yr between 2005 and 2024)
select ksic||'|'||mg cell, cc, appl_no, any_value(applicant) applicant, yr from t where mg is not null group by all
"""
con.execute(f"create temp table T as {q}")
tot = con.sql("select cell, yr, count(distinct cc||appl_no) n from T group by all").df()
krd = con.sql("select cell, yr, appl_no, applicant from T where cc='KR'").df()
print('loaded', round(time.time()-t,1), len(krd))
def norm(s):
    s = unicodedata.normalize('NFKC', html.unescape(s or '')).lower().replace('주식회사', '').replace('(주)', '').replace('㈜', '')
    return ''.join(ch for ch in s if ch.isalnum())
krd['a'] = krd.applicant.map({x: norm(x) for x in krd.applicant.unique()})
TP = tot.pivot_table(index='cell', columns='yr', values='n', aggfunc='sum', fill_value=0)
KP = krd.groupby(['cell','yr']).size().unstack(fill_value=0).reindex(TP.index).fillna(0)
for y in range(2005, 2025):
    if y not in TP: TP[y]=0
    if y not in KP: KP[y]=0
sm = lambda P,a,b: P[list(range(a,b+1))].sum(1)
DP = pd.read_parquet(os.path.join(PR, 'dev_preds_r3.parquet')); FP = pd.read_parquet(os.path.join(PR, 'FINAL_R3_t0_2021.parquet'))
rows=[]; cellrows=[]
for t0 in list(range(2010,2019))+[2021]:
    d = pd.DataFrame(dict(tot_W=sm(TP,t0-2,t0), kr_W=sm(KP,t0-2,t0), kr_fut=sm(KP,t0+1,t0+3))); d=d[d.tot_W>=50].reset_index()
    d['y']=np.log1p(d.kr_fut)-np.log1p(d.kr_W); d['D']=d.kr_fut-d.kr_W; d['h']=[int(hashlib.md5(c.encode()).hexdigest()[:8],16) for c in d.cell]
    d=d.sort_values(['y','D','h'],ascending=[False,False,True]).reset_index(drop=True); d['yb_re']=(np.arange(len(d))<round(0.1*len(d))).astype(int)
    fr = (FP if t0==2021 else DP[DP.t0==t0])[['cell','yb','p','p_base']]
    M = d.merge(fr, on='cell', how='outer', indicator=True)
    mism = int((M._merge!='both').sum()); M=M[M._merge=='both'].copy(); ybm=int((M.yb!=M.yb_re).sum())
    sub = krd[(krd.yr>=t0-2)&(krd.yr<=t0+3)&krd.cell.isin(set(M.cell))]
    da = sub.assign(s=np.where(sub.yr<=t0,-1,1)).groupby(['cell','a']).s.sum().groupby('cell').max()
    M['maxd']=M.cell.map(da).fillna(0); pos=M.yb==1
    M['sab']=pos&(M.D>0)&(M.maxd>=0.5*M.D)
    # 엄격 변형: 최대 기여 출원인이 증가분 전부 이상(≥1.0·D), 그리고 D>=5 한정
    M['sab100']=pos&(M.D>0)&(M.maxd>=1.0*M.D)
    r=dict(origin=t0,n=len(M),unmatched=mism,yb_mismatch=ybm,pos=int(pos.sum()),sab=int(M.sab.sum()),sab_rate=M.sab.sum()/pos.sum(),
           sab100_rate=M.sab100.sum()/pos.sum(),pos_krW_le1=float((pos&(M.kr_W<=1)).sum()/pos.sum()),pos_krW0=float((pos&(M.kr_W==0)).sum()/pos.sum()),
           pos_D_le3=float((pos&(M.D<=3)).sum()/pos.sum()),sab_rate_D5=float((M.sab&(M.D>=5)).sum()/(pos&(M.D>=5)).sum()),
           sab_rate_krW_ge2=float((M.sab&(M.kr_W>=2)).sum()/(pos&(M.kr_W>=2)).sum()),sab_rate_krW_le1=float((M.sab&(M.kr_W<=1)).sum()/(pos&(M.kr_W<=1)).sum()),
           elig_krW_le1=float((M.kr_W<=1).mean()))
    rows.append(r); print(r)
    M['t0']=t0; cellrows.append(M[pos][['t0','cell','kr_W','kr_fut','D','sab']])
R=pd.DataFrame(rows); R.to_csv(os.path.join(V,'v1_by_origin.csv'),index=False,encoding='utf-8-sig')
C=pd.concat(cellrows); C.to_csv(os.path.join(V,'v1_pos_cells.csv'),index=False,encoding='utf-8-sig')
print('sec',round(time.time()-t,1))

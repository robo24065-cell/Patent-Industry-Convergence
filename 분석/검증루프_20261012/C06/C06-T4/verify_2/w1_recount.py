# 반박 검증: NPS 키(biz6|정규화상호) 다사업장 비율·같은 키 새 사업장 사건·상호만 확장 사후진단·지점형 행 고용 비중 재계산
import sys, re, json, time; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT=B+r'\분석\모델경쟁_20260927\_shared\ext'
W=B+r'\분석\검증루프_20261012\C06\C06-T4'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
TC={str(x) for x in range(81,89)}
DEV=['201512','201612','201712','201812','201912','202012','202112','202212','202312']
REF={'201512':'20151130','201612':'20161130','201712':'20171130','201812':'20181130','201912':'20191130','202012':'20201231','202112':'20211231','202212':'20221231','202312':'20231231','202412':'20241231'}
def load(s):
    d=pd.read_parquet(EXT+rf'\firm_nps_wkpl_{s}.parquet',columns=['name','biz6','stat','adpt_dt','n_emp'])
    d=d[(d.stat.astype(str)=='1')&d.biz6.astype(str).str[3:5].isin(TC)].copy()
    d['nn']=d.name.map(norm); d['key']=d.biz6.astype(str)+'|'+d.nn
    d['ad']=d.adpt_dt.fillna('').astype(str).str.replace('-','').str[:8]
    return d
U=pd.read_parquet(W+r'\cache\B_units.parquet')
OUT={}; T0=time.time(); cache={}
for s0 in DEV:
    s1=str(int(s0[:4])+1)+'12'
    d0=cache.pop(s0,None)
    if d0 is None: d0=load(s0)
    d1=load(s1); cache[s1]=d1
    u=U[U.s0==s0]; ks=set(u.key)
    g0=d0.groupby('key').agg(E0=('n_emp','sum'),nw0=('ad','size'))
    # 전체 법인 키(E0>=5) 다사업장 비율
    allk=g0[g0.E0>=5]
    gu=g0.reindex(list(ks))
    r=dict(units=len(u), units_found=int(gu.E0.notna().sum()), E0_mismatch=int((gu.E0.values!=u.set_index('key').reindex(list(ks)).E0.values).sum()),
           multi_share_units=round(float((gu.nw0>1).mean()),5), multi_share_allcorp_E0ge5=round(float((allk.nw0>1).mean()),5), n_allcorp=len(allk))
    # 같은 키 새 사업장(적용일 창 안 + 사업장 수 증가)
    d1u=d1[d1.key.isin(ks)]
    nw1=d1u.groupby('key').size()
    newk=set(d1u[(d1u.ad>REF[s0])&(d1u.ad<=REF[s1])].key)
    nw0s=gu.nw0
    y=[k for k in newk if nw1.get(k,0)>nw0s.get(k,0)]
    r['pos_samekey']=len(y); r['pos_samekey_newad_only']=len(newk)
    # 사후: 상호만 같은 새 사업장(biz6 무관)
    nn_u=set(k.split('|',1)[1] for k in ks)
    newnn=set(d1[(d1.ad>REF[s0])&(d1.ad<=REF[s1])].nn)
    r['pos_name_only']=len(nn_u & newnn)
    # 사후: 지점형(같은 biz6, 정규화상호가 키 상호로 시작하고 더 김) 활성 행 — S0 시점, 키 고용 대비
    d0['b6']=d0.biz6.astype(str)
    ub=pd.DataFrame({'b6':[k.split('|',1)[0] for k in ks],'nn':[k.split('|',1)[1] for k in ks]})
    ub=ub[ub.nn.str.len()>=2]
    m=d0.merge(ub,on='b6',suffixes=('','_u'))
    br=m[(m.nn!=m.nn_u)&m.apply(lambda x: x.nn.startswith(x.nn_u),axis=1)]
    r['branchlike_sameb6_rows']=len(br); r['branchlike_sameb6_firms']=int(br.nn_u.nunique()); r['branchlike_sameb6_emp']=int(br.n_emp.sum())
    r['units_emp']=int(gu.E0.sum())
    OUT[s0]=r; print(s0,r,round(time.time()-T0),flush=True)
json.dump(OUT,open(W+r'\verify_2\w1_out.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)

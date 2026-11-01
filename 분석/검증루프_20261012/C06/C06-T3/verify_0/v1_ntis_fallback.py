# C06-T3 verify (반박 2차): PREREG §2.2 '없으면 NTIS 과제 지역' 대체 주소를 적용했을 때 M 시나리오 재채점
import pandas as pd,numpy as np,sys,io,json
from scipy.stats import rankdata
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
R=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
B=R+r'\분석\검증루프_20261012\C06\C06-T3'; CA=B+r'\cache'
pr=pd.read_parquet(R+r'\분석\모델경쟁_20260927\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet'); pr['bz']=pr.bz.astype(str).str.zfill(10); pr=pr.reset_index(drop=True)
c=pd.read_parquet(CA+r'\n6_cand.parquet'); p1=pd.read_parquet(CA+r'\n6_p1_pairs.parquet'); f=pd.read_parquet(CA+r'\n6_firm.parquet')
p1['window']=p1.ad.between(20220101,20231231); p1['known_le2021']=p1.known<=20211231
RG={'01':'서울','02':'부산','03':'대구','04':'인천','05':'광주','06':'대전','07':'울산','08':'경기','09':'강원','10':'충북','11':'충남','12':'전북','13':'전남','14':'경북','15':'경남','16':'제주','19':'세종'}
f['nt_sido']=f.region.map(RG)
c=c.merge(f[['bz','nt_sido']],on='bz',how='left')
# 경쟁 코드 시도 집합
c['comp_sidos']=c.comp_addrs.fillna('').map(lambda s:{x.split(' ')[0] for x in s.split(';') if x})
okc=c[~c.rl_other_only]
Mh=okc[okc.addr_sg_eq & ~okc.comp_same_sg]
miss=okc[okc.f_sido.isna()]
fb=miss[(miss.d_sido.notna())&(miss.d_sido==miss.nt_sido)&~miss.comp_same_sg]
fb_strict=fb[[d not in s for d,s in zip(fb.d_sido,fb.comp_sidos)]]
print('missing-addr cand pairs',len(miss),'firms',miss.bz.nunique(),'fallback pairs',len(fb),'strict',len(fb_strict))
y0=pr.y.values.astype(int); pc=pr.p_champ.values; pm=pr.p_m0.values
def auc(y,p):
    y=np.asarray(y).astype(int); r=rankdata(np.asarray(p,float)); n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
sk=lambda a,b:(a-b)/(1-b)
base=sk(auc(y0,pc),auc(y0,pm)); bzidx={b:i for i,b in enumerate(pr.bz)}
rank=rankdata(pc)/len(pc)
def apply(pairs,drop=True):
    flip=set(pairs[pairs.window].bz)&set(pr.bz[pr.y==0]); drp=set(pairs[pairs.known_le2021].bz) if drop else set()
    y=y0.copy(); y[[bzidx[b] for b in flip]]=1; keep=np.ones(len(y),bool); keep[[bzidx[b] for b in drp if b in bzidx]]=False
    return y,keep,flip,drp
rng=np.random.default_rng(777); boots=[rng.integers(0,len(y0),len(y0)) for _ in range(500)]
def ci(y,keep):
    out=[]
    for i in boots:
        a=sk(auc(y0[i],pc[i]),auc(y0[i],pm[i])); j=i[keep[i]]; out.append(sk(auc(y[j],pc[j]),auc(y[j],pm[j]))-a)
    return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
o={'base':base}
S={'M':pd.concat([p1,Mh]),'M+NTIS지역':pd.concat([p1,Mh,fb]),'M+NTIS지역_strict':pd.concat([p1,Mh,fb_strict]),'NTIS지역 대체분만(+P1)':pd.concat([p1,fb])}
for k,pairs in S.items():
    for drop in (True,False):
        y,keep,flip,drp=apply(pairs,drop); s1=sk(auc(y[keep],pc[keep]),auc(y[keep],pm[keep]))
        r=dict(flips=len(flip),drops=len(drp),pos=int(y[keep].sum()),skill=s1,dskill=s1-base,
               flip_pctile=float(np.mean([rank[bzidx[b]] for b in flip])) if flip else None)
        if drop: r['ci']=ci(y,keep)
        o[f'{k}|drop={drop}']=r; print(k,drop,r)
# 대체분 새 전환 기업의 점수 백분위
_,_,fM,_=apply(S['M']); _,_,fX,_=apply(S['M+NTIS지역'])
new=fX-fM; o['new_flips']=len(new); o['new_flip_pctile']=float(np.mean([rank[bzidx[b]] for b in new])) if new else None
print('new flips from fallback',len(new),o['new_flip_pctile'])
json.dump(o,open(B+r'\verify_0\v1_ntis_fallback_out.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)

# -*- coding: utf-8 -*-
# verify_1: F2(fam_n 상수) 독립 재계산. 입력 읽기만, 출력은 verify_1 아래만.
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
J=os.path.join
BASE=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석'
F22=J(BASE,'모델경쟁_20260927','T4_특허포기예측','R2','runs_final','F22')
W=J(BASE,'검증루프_20261012','C19','C19-T2')
OUT=J(W,'verify_1')
res={}
# 1) F2 fam_n 배열이 상수인지, 값이 P0 frame_cls 평균인지
p0c=np.load(J(W,'cache','famn','P0__frame_cls.npy')).astype('float64')
for fr in ['frame_cls','frame_haz','ev','ev_haz']:
    v=np.load(J(W,'cache','famn',f'F2__{fr}.npy'))
    res[f'F2_{fr}']=dict(n=int(len(v)),nuniq=int(len(np.unique(v))),val=float(v[0]))
res['P0_frame_cls_mean']=float(p0c.mean())
import pyarrow.parquet as pq
st=pq.read_table(J(F22,'frame_cls.parquet'),columns=['fam_n']).to_pandas().fam_n.values.astype('float64')
res['stored_frame_cls_famn_mean']=float(st.mean()); res['stored_frame_cls_famn_std']=float(st.std())
ev=pd.read_parquet(J(F22,'ev.parquet'),columns=['pid','acd','y','fam_n'])
res['ev_famn_std']=float(ev.fam_n.astype(float).std()); res['ev_famn_pos_share']=float((ev.fam_n>0).mean())
row_h=pd.read_parquet(J(F22,'ev_haz.parquet'),columns=['row']).row.values
y=ev.y.values.astype(int); N=len(y)
def comb_haz(h):
    lg=pd.Series(np.log1p(-np.clip(h,1e-6,1-1e-6))).groupby(row_h).sum()
    P=np.zeros(N); P[lg.index.values]=1-np.exp(lg.values); return P
def build(d, kc='cls_99f270b3cd', kh='haz_0568d4297b', kl='logit_3008b8b195', kg='hgb_b4bd0fa46a'):
    zc=np.mean([np.load(J(d,f'{kc}_b{k}.npy')) for k in range(3)],0); pc=(1/(1+np.exp(-zc))).astype('float32')
    zh=np.mean([np.load(J(d,f'{kh}_b{k}.npy')) for k in range(3)],0); ph=comb_haz(1/(1+np.exp(-zh)))
    pl=np.load(J(d,f'{kl}.npy')); pg=np.load(J(d,f'{kg}.npy'))
    lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    z=(0.2*lg(pc)+0.4*lg(ph)+0.2*lg(pl)+0.2*lg(pg))/1.0
    return 1/(1+np.exp(-z)), dict(cls=pc,haz=ph,logit=pl,hgb=pg)
pO,cO=build(J(F22,'pred')); pF,cF=build(J(W,'cache','runs','F2','pred'))
PB=np.load(J(F22,'pred','base_46526c56d8.npy'))
# 내 AUC (Mann-Whitney, 타이 0.5)
def prep(s):
    o=np.argsort(s,kind='mergesort'); ss=s[o]
    g=np.r_[0,np.cumsum(ss[1:]!=ss[:-1])]
    return o,g,g[-1]+1
def wauc(P,w):
    o,g,G=P
    wp=np.bincount(g,weights=(w*y)[o],minlength=G); wn=np.bincount(g,weights=(w*(1-y))[o],minlength=G)
    cn=np.cumsum(wn)-wn
    return float((wp*(cn+0.5*wn)).sum()/(wp.sum()*wn.sum()))
PO,PF,PBp=prep(pO),prep(pF),prep(PB)
one=np.ones(N)
aO,aF,aB=wauc(PO,one),wauc(PF,one),wauc(PBp,one)
sk=lambda a,b:(a-b)/(1-b)
res.update(auc_off=aO,auc_F2=aF,auc_M0=aB,skill_off=sk(aO,aB),skill_F2=sk(aF,aB),d_skill=sk(aF,aB)-sk(aO,aB),d_auc=aF-aO)
res['max_diff_vs_saved_pred_F2']=float(np.abs(pF-np.load(J(W,'cache','pred_F2.npy'))).max())
res['max_diff_vs_saved_pred_OFF']=float(np.abs(pO-np.load(J(W,'cache','pred_OFFICIAL.npy'))).max())
res['comp_auc_F2']={k:wauc(prep(v.astype('float64')),one) for k,v in cF.items()}
res['comp_auc_off']={k:wauc(prep(v.astype('float64')),one) for k,v in cO.items()}
res['corr_pred_off_F2']=float(np.corrcoef(pO,pF)[0,1])
from scipy.stats import spearmanr
res['spearman_off_F2']=float(spearmanr(pO,pF).correlation)
# 군집 짝 부트스트랩 (다른 seed)
cl=np.where(ev.acd.values=='NA','P'+ev.pid.astype(str).values,ev.acd.values)
codes,u=pd.factorize(cl); K=len(u); res['K']=int(K)
# 결측 acd 다른 표기 점검
res['acd_null']=int(pd.isna(ev.acd).sum())
for seed,B in [(777,600),(20261017,400)]:
    rng=np.random.default_rng(seed); D=[];DA=[]
    for _ in range(B):
        w=np.bincount(rng.integers(0,K,K),minlength=K).astype(float)[codes]
        a1,a0,ab=wauc(PF,w),wauc(PO,w),wauc(PBp,w)
        D.append(sk(a1,ab)-sk(a0,ab)); DA.append(a1-a0)
    res[f'boot_seed{seed}_B{B}']=dict(d_skill_ci=[float(np.percentile(D,2.5)),float(np.percentile(D,97.5))],d_auc_ci=[float(np.percentile(DA,2.5)),float(np.percentile(DA,97.5))],d_skill_sd=float(np.std(D)))
# 상위 k
def topk(p,k):
    o=np.argsort(-p,kind='stable')[:k]; return o,int(y[o].sum())
oO,hO=topk(pO,1000); oF,hF=topk(pF,1000)
res['top1000']=dict(off=hO,F2=hF,overlap=int(len(np.intersect1d(oO,oF))))
k1=int(round(0.01*N)); res['top1pct']=dict(k=k1,off=topk(pO,k1)[1],F2=topk(pF,k1)[1])
json.dump(res,open(J(OUT,'r1_recompute.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))

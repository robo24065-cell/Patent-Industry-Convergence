# -*- coding: utf-8 -*-
"""r3oa 10 (05의 v2, 2026-09-29 재시도): 균형 보강 포함 + 변형(company/KR) 검정 + 셀 고유 차이(메인그룹−KSIC)·변형 증분(변형 − ALL) 편상관 추가.
 연계율 + 개발 기간 선행 검정 (T3 셀, 원점 t0=2006..2016, 결과 KR 출원 t0+1..t0+3 <= 2019).
🔒 최종시험 구간(2020~) 결과는 계산하지 않는다: 패널은 T3 R2 cache_dev(출원연도 <=2019)만 읽고, 레이블은 t0+3<=2019 확인.
x = 메인그룹 논문 점유(전 세계 전체 논문 대비) 성장: log(Σt0-2..t0) − log(Σt0-5..t0-3)  — 논문은 t0년까지만.
y = log((KR t0+1..t0+3 + 1)/(KR t0-2..t0 + 1))  (T3 레이블과 동일)
통제 = 셀 KR 모멘텀, 셀 5국 모멘텀, KSIC KR 특허 모멘텀, KSIC 논문 점유 성장(기존 science_ KSIC 수준 x).
순위 편상관(원점별) + 순열 귀무(메인그룹→x 대응을 섞음, 200회)."""
import pandas as pd, numpy as np, os, sys, json
from scipy.stats import rankdata
W=os.path.dirname(os.path.abspath(__file__)); EXT=os.path.dirname(W)
VAR=sys.argv[1] if len(sys.argv)>1 else 'ALL'
MODE=sys.argv[2] if len(sys.argv)>2 else 'app'   # app: 출원연도(결과 <=2019) / pub: 공개연도(T3 R3 최종시험 방식, 결과 <=2021, t0<=2018)
MAXY=2019 if MODE=='app' else 2021; T0S=range(2006,2017) if MODE=='app' else range(2006,2019)
PAN=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R2\cache_dev\panel_ccy.parquet'
if MODE=='pub':  # T3 R3 개발 캐시(공개연도 <=2021), 읽기 전용
    PAN=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R3\cache_pub\panel_ccy.parquet'
P=pd.read_parquet(PAN,columns=['ksic','mg','cc','year','n'])
assert P.year.max()<=MAXY
Y0=2000; YR=np.arange(2000,MAXY+1)
P['cell']=P.ksic+'|'+P.mg
cells=np.array(sorted(P.cell.unique())); cid={c:i for i,c in enumerate(cells)}
CC=['KR','US','CN','JP','EP']
A=np.zeros((len(cells),5,len(YR)))
A[P.cell.map(cid).values,P.cc.map({c:i for i,c in enumerate(CC)}).values,P.year.values-Y0]=P.n.values
ks=np.array([c.split('|')[0] for c in cells]); mgs=np.array([c.split('|')[1] for c in cells])
def ws(a,t0,lo,hi):
    assert t0+hi<=MAXY, '🔒'
    return a[...,t0+lo-Y0:t0+hi-Y0+1].sum(-1)
KR=A[:,0,:]; TOT=A.sum(1)
# KSIC KR 특허 (셀 합이 아닌 KSIC 단위 중복 제거가 이상적이나 여기선 셀 합 근사)
ksu=np.unique(ks); kidx={k:i for i,k in enumerate(ksu)}
KK=np.zeros((len(ksu),len(YR)))
np.add.at(KK,[ks_i for ks_i in [np.array([kidx[k] for k in ks])]][0],KR)
# 논문
OA=pd.read_parquet(os.path.join(EXT,'r3oa_mg_year_%s.parquet'%VAR))
TOTW=pd.read_parquet(os.path.join(EXT,'science_openalex_country_total_year.parquet'))
tw=TOTW[TOTW.country=='ALL'].set_index('year').n_works_total
if VAR=='company':
    ex=pd.read_parquet(os.path.join(EXT,'science_openalex_ksic_extra_year.parquet'))
    tw=ex[(ex.ksic=='TOTAL')&(ex.variant=='company_ALL')].set_index('year').n_works
elif VAR!='ALL':
    tw=TOTW[TOTW.country==VAR].set_index('year').n_works_total
pw=OA.pivot_table(index='mg',columns='year',values='n_works',aggfunc='first')
OAA=pd.read_parquet(os.path.join(EXT,'r3oa_mg_year_ALL.parquet'))
pwA=OAA.pivot_table(index='mg',columns='year',values='n_works',aggfunc='first')
KW=pd.read_parquet(os.path.join(EXT,'science_openalex_ksic_country_year.parquet'))
kw=KW[KW.country=='ALL'].pivot_table(index='ksic',columns='year',values='n_works')
twA=TOTW[TOTW.country=='ALL'].set_index('year').n_works_total
EXK=pd.read_parquet(os.path.join(EXT,'science_openalex_ksic_extra_year.parquet'))
kwc=EXK[EXK.variant=='company_ALL'].pivot_table(index='ksic',columns='year',values='n_works')
twc=EXK[(EXK.ksic=='TOTAL')&(EXK.variant=='company_ALL')].set_index('year').n_works
def pgrowth(row,tot,t0):
    a=sum(row.get(y,0) for y in range(t0-2,t0+1)); b=sum(row.get(y,0) for y in range(t0-5,t0-2))
    ta=sum(tot[y] for y in range(t0-2,t0+1)); tb=sum(tot[y] for y in range(t0-5,t0-2))
    return np.log((a+1)/ta)-np.log((b+1)/tb), a, b
def resid(y,X):
    X=np.column_stack([np.ones(len(y))]+X); b=np.linalg.lstsq(X,y,rcond=None)[0]; return y-X@b
def pcorr(x,y,C):
    rx=resid(rankdata(x),[rankdata(c) for c in C]); ry=resid(rankdata(y),[rankdata(c) for c in C])
    return np.corrcoef(rx,ry)[0,1]
rng=np.random.default_rng(7)
res=[]; link=[]
for t0 in T0S:
    past=ws(TOT,t0,-2,0); el=past>=50
    has=np.isin(mgs,pw.index)
    link.append(dict(t0=t0,elig_cells=int(el.sum()),covered=int((el&has).sum()),share=float((el&has).mean()/el.mean()),
                     pat5_share=float(past[el&has].sum()/past[el].sum()),kr_share=float(ws(KR,t0,-2,0)[el&has].sum()/max(ws(KR,t0,-2,0)[el].sum(),1)),
                     n_ksic_covered=int(len(set(ks[el&has]))),n_mg=int(len(set(mgs[el&has])))))
    m=el&has
    y=np.log((ws(KR,t0,1,3)+1)/(ws(KR,t0,-2,0)+1))[m]
    krm=np.log(ws(KR,t0,-2,0)+1)-np.log(ws(KR,t0,-5,-3)+1)
    tm=np.log(ws(TOT,t0,-2,0)+1)-np.log(ws(TOT,t0,-5,-3)+1)
    ki=np.array([kidx[k] for k in ks])
    km=(np.log(ws(KK,t0,-2,0)+1)-np.log(ws(KK,t0,-5,-3)+1))[ki]
    mg_x={g:pgrowth(pw.loc[g].to_dict(),tw,t0)[0] for g in set(mgs[m])}
    mgA={g:pgrowth(pwA.loc[g].to_dict(),twA,t0)[0] for g in set(mgs[m])}
    x=np.array([mg_x[g] for g in mgs[m]])
    kx_map={k:pgrowth(kw.loc[k].to_dict(),twA,t0)[0] for k in set(ks[m])}
    kx=np.array([kx_map[k] for k in ks[m]])
    C1=[krm[m],tm[m],km[m]]; C2=C1+[kx]
    r1=pcorr(x,y,C1); r2=pcorr(x,y,C2); rk=pcorr(kx,y,C1)
    xA=np.array([mgA[g] for g in mgs[m]])
    r_inc=pcorr(x,y,C2+[xA]) if VAR!='ALL' else np.nan      # 변형 증분: ALL 메인그룹 성장·KSIC 논문까지 통제
    kxc=np.array([pgrowth(kwc.loc[k].to_dict(),twc,t0)[0] if k in kwc.index else 0.0 for k in ks[m]])
    r_kc=pcorr(x,y,C1+[kxc])   # 기존 KSIC 기업 논문 성장(science_ company_ALL)을 통제한 셀 수준 증분
    r_diff=pcorr(x-kx,y,C1)                                  # 셀 고유: 메인그룹 성장 − KSIC 성장
    # (보조 가설) 과학 집약도: 메인그룹 논문 수준 대비 셀 5국 특허 수준 log((논문 t0-2..t0 +1)/(특허 t0-2..t0 +1))
    lv={g:pgrowth(pw.loc[g].to_dict(),tw,t0)[1] for g in set(mgs[m])}
    si=np.log(np.array([lv[g] for g in mgs[m]])+1)-np.log(ws(TOT,t0,-2,0)[m]+1)
    r3=pcorr(si,y,C1+[np.log(ws(TOT,t0,-2,0)[m]+1)])
    # (변형 전용) 변형 비중 수준: log(변형 논문 t0-2..t0 +1) − log(ALL 논문 t0-2..t0 +1), 예: 기업 논문 비중(상업화 근접도)
    if VAR!='ALL':
        lvA={g:pgrowth(pwA.loc[g].to_dict(),twA,t0)[1] for g in set(mgs[m])}
        sh=np.array([np.log(lv[g]+1)-np.log(lvA[g]+1) for g in mgs[m]])
        r_sh=pcorr(sh,y,C1+[np.log(np.array([lvA[g] for g in mgs[m]])+1)])
    else: r_sh=np.nan
    # 순열: 메인그룹 단위로 x를 섞음
    glist=np.array(sorted(mg_x)); gx=np.array([mg_x[g] for g in glist]); gi=pd.Index(glist).get_indexer(mgs[m])
    null1=[];null2=[]
    for _ in range(200):
        xp=rng.permutation(gx)[gi]; null1.append(pcorr(xp,y,C1)); null2.append(pcorr(xp,y,C2))
    null1=np.array(null1); null2=np.array(null2)
    # 부상(상위10%) 여부별 x 평균
    k=int(round(0.1*len(y))); top=np.zeros(len(y),bool); top[np.argsort(-y,kind='stable')[:k]]=True
    res.append(dict(t0=t0,n=int(m.sum()),n_mg=len(glist),pc_mg=r1,null_sd=null1.std(),p_perm=float((null1>=r1).mean()),
                    pc_mg_ctrl_ksicpaper=r2,p_perm_ctrl=float((null2>=r2).mean()),pc_ksicpaper=rk,pc_sci_intensity=r3,pc_inc_over_ALL=r_inc,pc_diff_mg_minus_ksic=r_diff,pc_variant_share_level=r_sh,pc_ctrl_ksic_company=r_kc,
                    x_mean_breakout=float(x[top].mean()),x_mean_rest=float(x[~top].mean())))
    print(res[-1],flush=True)
R=pd.DataFrame(res); L=pd.DataFrame(link)
R.to_csv(os.path.join(EXT,'r3oa_test_leadlag_v2_%s%s.csv'%(VAR,'' if MODE=='app' else '_pub')),index=False)
L.to_csv(os.path.join(EXT,'r3oa_link_cells_v2_%s%s.csv'%(VAR,'' if MODE=='app' else '_pub')),index=False)
print(L.to_string()); print(R.to_string())
print('mean pc_mg',R.pc_mg.mean(),'pos',(R.pc_mg>0).sum(),'/',len(R),'sig',(R.p_perm<0.05).sum())
print('mean pc_mg|ksicpaper',R.pc_mg_ctrl_ksicpaper.mean(),'pos',(R.pc_mg_ctrl_ksicpaper>0).sum(),'sig',(R.p_perm_ctrl<0.05).sum())
print('mean pc_ksicpaper',R.pc_ksicpaper.mean())
print('mean pc_sci_intensity',R.pc_sci_intensity.mean(),'pos',(R.pc_sci_intensity>0).sum())
print('mean pc_inc_over_ALL',R.pc_inc_over_ALL.mean(),'pos',(R.pc_inc_over_ALL>0).sum())
print('mean pc_diff_mg_minus_ksic',R.pc_diff_mg_minus_ksic.mean(),'pos',(R.pc_diff_mg_minus_ksic>0).sum())
print('early(2006-11) mean pc_mg',R[R.t0<=2011].pc_mg.mean(),'late(2012-16)',R[R.t0>=2012].pc_mg.mean())
print('mean pc_variant_share_level',R.pc_variant_share_level.mean(),'pos',(R.pc_variant_share_level>0).sum())
print('mean pc_ctrl_ksic_company',R.pc_ctrl_ksic_company.mean(),'pos',(R.pc_ctrl_ksic_company>0).sum())

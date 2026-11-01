from s2lib import *
import sys, itertools
D=data(True); b=0.6527
# 규모×모멘텀 비선형 항 (로지스틱용)
D['kr_lvl2']=D.kr_lvl**2; D['lvl_x_mom']=D.kr_lvl*D.kr_mom; D['lvl_x_us']=D.kr_lvl*D.US_mom; D['lvl_x_ep']=D.kr_lvl*D.EP_mom
D['lvl_x_dd']=D.kr_lvl*D.kr_drawdown; D['lvl_x_sh']=D.kr_lvl*D.kr_share_rel_ksic; D['zero']=(D.kr_W==0).astype(float); D['le2']=(D.kr_W<=2).astype(float)
INT=['kr_lvl2','lvl_x_mom','lvl_x_us','lvl_x_ep','lvl_x_dd','lvl_x_sh','zero','le2']
which=sys.argv[1:]
if 'v36' in which:
    both(logit(PRUNE),D,'logit PRUNE'); both(logit(PRUNE+INT),D,'logit PRUNE+INT')
    R,P,Rx=both(ens(PRUNE+INT,ALL2,ALL2),D,'v36'); sx=H.summarize(Rx)
    H.record('v36','[2단계] v30의 로지스틱 성분에 규모 비선형·상호작용 8항(kr_lvl², 규모×KR모멘텀·US·EP모멘텀·낙폭·상대점유율, KR=0·≤2 지시자) 추가',
      '가장 큰 레버였던 규모×모멘텀 비선형(v1 +0.089)을 선형 성분도 직접 표현하게 함',R,P,b,kept='?',note=f'추가원점(2010/12/14) {sx["auc"]:.4f} ({sx["auc_by_t0"]})')
if 'v37' in which:
    # 성분 가중치를 추가원점(2010/12/14)에서만 고르고 → 검증원점에서 정직하게 평가
    comps=[('lg',logit(PRUNE+INT) if 'int' in which else logit(PRUNE)),('xg',xgbc(ALL2,XP,rounds=300)),('nb',nbfp(ALL2))]
    def comp_preds(val):
        out={}
        for v in val:
            tr=D[D.t0<=v-3]; te=D[D.t0==v]
            out[v]=(te,[rankdata(c(tr,te))/len(te) for _,c in comps])
        return out
    from sklearn.metrics import roc_auc_score
    X=comp_preds(EXTRA_T0); V=comp_preds(L.VAL_T0)
    grid=[w for w in itertools.product([0,1,2,3],repeat=3) if sum(w)>0]
    def score(C,w): return np.mean([roc_auc_score(te.yb,np.average(ps,axis=0,weights=w)) for te,ps in C.values()])
    sc=sorted([(score(X,w),w) for w in grid],reverse=True)
    print('추가원점 상위 가중치',sc[:6]); wbest=sc[0][1]
    print('동일가중 추가원점',score(X,(1,1,1)),'검증',score(V,(1,1,1)))
    print('선택가중',wbest,'검증',score(V,wbest))
    R,P=H.rolling(ens(PRUNE+INT if 'int' in which else PRUNE,ALL2,ALL2,w=wbest),D)
    H.record('v37','[2단계] 3성분 순위평균 가중치를 추가원점(2010/12/14)에서만 격자(0~3)로 고르고 검증원점에서 평가'+(' (로지스틱 v36 상호작용 포함)' if 'int' in which else ''),
      '메타학습(v34) 계수상 로지스틱>XGB>음이항 → 가중을 두되 검증 과적합 없이',R,P,b,kept='?',note=f'선택 가중(lg,xg,nb)={wbest}; 추가원점 {sc[0][0]:.4f} vs 동일가중 {score(X,(1,1,1)):.4f}')

from models import *
D=enrich(H.data())
xg=xgbc(ALL2,XP,rounds=300); lg=logit(PRUNE)
def ens(tr,te,lgf=lg,xgf=xg): return rankavg(lgf(tr,te), xgf(tr,te), nbfp(ALL2)(tr,te))
R,P=H.rolling(lambda tr,te: rankavg(multi_thr(lg)(tr,te), multi_thr(xg)(tr,te), nbfp(ALL2)(tr,te)),D)
H.record('v19','v17 앙상블의 로지스틱·XGB를 상위 5/10/20/30% 다중 임계 레이블로 각각 학습해 평균','경계 근처 레이블 잡음을 평활하면 순위가 안정',R,P,BASE_AUC,kept='?')
for half in [2,4]:
    R,P=H.rolling(lambda tr,te: ens(tr,te,lgf=logit_w(PRUNE,half)),D); print('half',half,round(H.summarize(R)['auc'],4))
H.record('v20','v17 앙상블의 로지스틱에 최근 원점 가중(반감기 4년)','구조변화(CN 부상 등)로 최근 원점이 더 유효',R,P,BASE_AUC,kept='?',note='반감기 2년도 시험(위 로그)')
def fp21(tr,te):
    tr=tr.copy(); te=te.copy()
    tr['sub']=tr.mg.str[:4]; te['sub']=te.mg.str[:4]
    for k in ['ksic','sub','mg']:
        a,b=te_enc(tr,te,k); tr['te_'+k]=a; te['te_'+k]=b
    c=PRUNE+['te_ksic','te_sub','te_mg']; c2=ALL2+['te_ksic','te_sub','te_mg']
    return rankavg(logit(c)(tr,te), xgbc(c2,XP,rounds=300)(tr,te), nbfp(ALL2)(tr,te))
R,P=H.rolling(fp21,D); H.record('v21','v17 + 과거 원점의 산업(KSIC)·서브클래스·메인그룹별 breakout 빈도 타깃인코딩(베이즈 축소, prior=20)','부상 성향은 산업·기술분야 고유 특성으로 지속',R,P,BASE_AUC,kept='?',note='인코딩은 학습 원점(≤v-3)만 사용. 학습셋 내부 자기포함 인코딩이라 학습 과적합 가능')

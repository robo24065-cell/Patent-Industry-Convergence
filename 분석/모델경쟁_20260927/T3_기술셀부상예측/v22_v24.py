from models import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing
D=enrich(H.data())
xg=xgbc(ALL2,XP,rounds=300); lg=logit(PRUNE)
# v22: OOF ksic 타깃인코딩
def fp22(tr,te):
    tr=tr.copy(); te=te.copy()
    enc_tr=oof_origin(tr, lambda a,b: te_enc(a,b,'ksic')[1])
    tr['te_ksic']=enc_tr; te['te_ksic']=te_enc(tr,te,'ksic')[1]
    return rankavg(logit(PRUNE+['te_ksic'])(tr,te), xgbc(ALL2+['te_ksic'],XP,rounds=300)(tr,te), nbfp(ALL2)(tr,te))
R,P=H.rolling(fp22,D); H.record('v22','v21 수정: KSIC 타깃인코딩만, 학습셋은 원점-OOF(결과창 겹치는 ±2 원점 제외)로 인코딩','v21 악화 원인=학습셋 자기포함 인코딩 누수; OOF면 산업 성향 신호만 남음',R,P,BASE_AUC,kept='?')
# v23: 통계 시계열 — 셀별 감쇠 추세 지수평활(Holt damped)로 KR 3년 합 예측 → 예측 성장률
def holt_score(df):
    P_=L.load(); kr=P_['A']['n'][:,0,:]; idx=np.searchsorted(P_['cells'],df.cell.values); t0=int(df.t0.iloc[0])
    X=np.log1p(kr[idx, t0-7-L.Y0:t0-L.Y0+1]) if t0-7>=L.Y0 else np.log1p(kr[idx,:t0-L.Y0+1])
    # 벡터화 Holt damped (alpha=.5,beta=.2,phi=.8) — 고정 파라미터
    a,b,phi=0.5,0.2,0.8; lvl=X[:,0].copy(); tr_=np.zeros(len(X))
    for j in range(1,X.shape[1]):
        prev=lvl.copy(); lvl=a*X[:,j]+(1-a)*(prev+phi*tr_); tr_=b*(lvl-prev)+(1-b)*phi*tr_
    fc=np.log(np.sum([np.expm1(np.maximum(lvl+sum(phi**i for i in range(1,h+1))*tr_,0)) for h in (1,2,3)],axis=0)+1)
    return fc-np.log(df.kr_W.values+1)
D['holt_g']=np.nan
for t,g in D.groupby('t0'): D.loc[g.index,'holt_g']=holt_score(g)
def fp23(tr,te):
    return logit(['holt_g','kr_lvl'])(tr,te)
R,P=H.rolling(fp23,D); H.record('v23','통계 시계열 단독: 셀별 Holt 감쇠추세 지수평활(log1p KR, α.5 β.2 φ.8)로 3년 예측성장 + KR 규모 로지스틱','시계열 평활 추세가 단순 모멘텀보다 잡음에 강함',R,P,BASE_AUC,kept='N',note='다른 모형족 비교용')
R,P=H.rolling(lambda tr,te: rankavg(lg(tr,te), xg(tr,te), nbfp_het(ALL2)(tr,te)),D)
H.record('v24','v17의 음이항 성분을 KR 과거규모 5구간별 분산(alpha) 추정으로 교체','소규모 셀일수록 과산포가 달라 꼬리확률 보정 필요',R,P,BASE_AUC,kept='?')

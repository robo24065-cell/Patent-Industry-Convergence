from models import *
D=H.data()
ALL=MOM+FOR+STR+GRP
V=[('v5','XGBoost 분류(깊이3, 300라운드) 전체 피처 70여개','비선형 상호작용(규모×모멘텀×해외)',xgbc(ALL)),
   ('v6','XGBoost 회귀(y_growth 연속값) 전체 피처','연속 성장률 학습이 희소 레이블보다 정보량 많음',xgbc(ALL,target='y')),
   ('v7','XGBoost 순위학습(rank:pairwise, t0별 그룹, 레이블=y 분위 10등급)','순위 목적함수가 AUC와 직접 정렬',None)]
def rankfp(tr,te):
    tr=tr.sort_values('t0'); grp=tr.groupby('t0').size().values
    lab=tr.groupby('t0').y.transform(lambda s: np.floor(s.rank(pct=True)*9.999)).values
    d=xgb.DMatrix(tr[ALL].values,label=lab); d.set_group(grp)
    p=dict(objective='rank:pairwise',max_depth=3,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,nthread=2,verbosity=0)
    m=xgb.train(p,d,300); return m.predict(xgb.DMatrix(te[ALL].values))
V[2]=(V[2][0],V[2][1],V[2][2],rankfp)
for v,c,h,fp in V:
    R,P=H.rolling(fp,D); H.record(v,c,h,R,P,BASE_AUC,kept='?')

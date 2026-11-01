from models import *
D=enrich(H.data())
for C in [0.01,0.1,10]:
    R,_=H.rolling(logit(PRUNE,C=C),D); print('C',C,round(H.summarize(R)['auc'],4))
def fp16(tr,te): return rankavg(logit(PRUNE)(tr,te), xgbc(ALL2,XP,rounds=300)(tr,te))
R,P=H.rolling(fp16,D); H.record('v16','앙상블: v13 로지스틱(pruned) + v11 XGB 순위평균','선형·트리 상보성(v8에서 +0.009 확인)을 개선 피처셋에 적용',R,P,BASE_AUC,kept='?')
def fp17(tr,te): return rankavg(logit(PRUNE)(tr,te), xgbc(ALL2,XP,rounds=300)(tr,te), nbfp(ALL2)(tr,te))
R,P=H.rolling(fp17,D); H.record('v17','앙상블: v16 + 음이항 꼬리확률 구조모형(ALL2 피처)','계수잡음 구조모형이 소규모 셀 순위를 보완',R,P,BASE_AUC,kept='?')
# v18: 학습 표본 확대 (tot_W>=20 학습, 평가는 >=50 유지)
D20=enrich(H.data(20))
def roll_expand(fp):
    res=[];preds=[]
    for v in L.VAL_T0:
        tr=D20[D20.t0<=v-3]; te=D[D.t0==v]
        p=fp(tr,te); m=L.evaluate(te.y.values,p,te.yb.values); m['t0']=v; res.append(m)
        preds.append(pd.DataFrame({'cell':te.cell.values,'t0':v,'p':p,'y':te.y.values,'yb':te.yb.values}))
    return pd.DataFrame(res),pd.concat(preds,ignore_index=True)
R,P=roll_expand(fp16); H.record('v18','v16 앙상블을 확대 학습표본(5개국 합 ≥20, breakout은 그 표본 내 상위10%)으로 학습, 평가는 ≥50 유지','학습 표본 2~3배 확대로 분산 감소',R,P,BASE_AUC,kept='?',note='학습 레이블 정의가 평가 표본과 약간 다름(적격 모집단 차이)')

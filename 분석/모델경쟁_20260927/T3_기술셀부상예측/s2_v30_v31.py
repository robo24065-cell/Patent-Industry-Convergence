from s2lib import *
D=add_flow(data(True)); b=0.6527
print(D[FLOW].describe().T[['count','mean','std']].round(3))
R,P,Rx=both(ens(PRUNE,ALL2,ALL2),D,'v30 v17(중립)')
sx=H.summarize(Rx)
H.record('v30','[2단계] v17(선형+XGB+음이항, 타깃인코딩 없음)을 중립 레이블로 재측정','v29에서 KSIC 타깃인코딩 이득(+0.0013)이 동률깨기 인위성과 섞여 사라짐 → 더 단순한 v17이 동급',R,P,b,kept='Y',note=f'추가원점(2010/12/14) {sx["auc"]:.4f} ({sx["auc_by_t0"]}), 기준선 0.6293')
R,P,Rx=both(ens(PRUNE+FLOW,ALL2+FLOW,ALL2+FLOW),D,'v31 v30+우선권흐름')
sx=H.summarize(Rx)
H.record('v31','[2단계] v30 + 국가 간 우선권 흐름 13피처(한국발 해외출원 KR우선권 수준·모멘텀·KR대비 비율·US관청 KR우선권 모멘텀, KR관청 US/JP/EU/CN 우선권 유입 모멘텀·비중) 3성분 모두',
 '한국 출원인의 해외출원 확대(국제화)와 국가별 기술유입은 국내 출원 부상의 선행지표',R,P,b,kept='?',note=f'추가원점 {sx["auc"]:.4f} ({sx["auc_by_t0"]})')
# 흐름 피처 단일 추가 기여(로지스틱 v13 기준)
R0,_,R0x=both(logit(PRUNE),D,'logit PRUNE')
for f in ['out_kr_lvl','out_kr_mom','out_kr_ratio','out_kr_ratio_chg','out_us_kr_mom','in_US_mom','in_JP_mom','in_EU_mom','in_CN_mom','in_US_sh','in_JP_sh']:
    both(logit(PRUNE+[f]),D,'  +'+f)

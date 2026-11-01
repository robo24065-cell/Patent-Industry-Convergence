from models import *
D=add_lead(enrich(H.data()))
R,_=H.rolling(logit(PRUNE+LEAD),D); print('logit only',H.summarize(R))
R,P=H.rolling(best_fp2(PRUNE+LEAD,ALL2+LEAD,ALL2+LEAD),D)
H.record('v27','v22 + 선행시차 피처(US/JP/EP/CN 시차3 성장, 4개국 동일가중 평균성장 시차0~3, 성장국 수, 동일가중갭)',
 '선행시차 분석: JP·EP는 시차3까지 유의, CN은 시차0~1만 → 긴 시차·동일가중(CN 규모 지배 제거) 신호 추가',R,P,BASE_AUC,kept='?')

from models import *
D=enrich(H.data())
R,P=H.rolling(logit(PRUNE),D); H.record('v13','로지스틱 v12에서 기여 음(-)인 JP·CN·출원인포트폴리오·서브클래스 피처 제거(ablation 기반)','불안정한 국가 피처(CN 급팽창·JP 감소 추세 드리프트) 제거로 일반화 향상',R,P,BASE_AUC,kept='?',note='ablation이 검증 구간 기반이라 선택 편향 가능')
cols=MOM+FOR+NEW+STR+GRP
D2=rank_norm(D,cols); rc=[c+'_rn' for c in cols]
R,P=H.rolling(logit(rc),D2); H.record('v14','로지스틱: 모든 피처(구조·그룹 포함)를 t0 단면 순위→정규분위 변환','원점별 분포 드리프트(CN 폭증, 신규출원인 좌측절단)를 제거하면 구조 피처가 도움됨',R,P,BASE_AUC,kept='?')
R,P=H.rolling(xgbc(rc,XP,rounds=300),D2); H.record('v15','XGB(v10 설정): t0 단면 순위정규화 피처 전체','트리도 드리프트 제거로 원점 간 일관성 향상',R,P,BASE_AUC,kept='?')

from models import *
D=enrich(H.data())
print(D[NEW].describe().T[['count','mean','std']].round(3))
R,P=H.rolling(xgbc(ALL,XP,rounds=300),D); H.record('v10','XGB 하이퍼파라미터 격자(깊이2/3/4/6×라운드×eta×mcw 32조합) → 깊이2·300라운드·eta0.05·mcw5','얕은 트리가 소표본 원점(4~8개)에 덜 과적합',R,P,BASE_AUC,kept='?',note='격자 전체 0.753~0.766 평탄(grid_v10.csv) — 하이퍼파라미터는 주 레버 아님; 검증에서 선택했으므로 낙관 편향 있음')
R,P=H.rolling(xgbc(ALL2,XP,rounds=300),D); H.record('v11','v10 + 출원인 포트폴리오 모멘텀·대형출원인 비중·1위 비중·공동분류 이웃셀 모멘텀·서브클래스 모멘텀·장기 KR 이력(낙폭)·상대 KR 점유율(산업/서브클래스/메인그룹 기준)','셀 출원인들의 전사적 성장, 기술이웃 확산, 과거 대비 낙폭 반등, 타국 대비 KR 과소대표(추격)가 부상 선행',R,P,BASE_AUC,kept='?')
R,P=H.rolling(logit(MOM+FOR+NEW),D); H.record('v12','로지스틱: v2 피처 + v11 신규 피처','신규 피처의 선형 효과 확인(트리 대비)',R,P,BASE_AUC,kept='?')

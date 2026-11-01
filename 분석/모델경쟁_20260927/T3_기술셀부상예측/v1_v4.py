from models import *
D=H.data()
V=[('v1','로지스틱: KR 다중시차 모멘텀·가속도·수준 9개','평균회귀 강도는 규모·가속도에 따라 다름',logit(MOM)),
   ('v2','v1 + 해외 4개국 동일셀 모멘텀(시차0/1/2)·수준·KR점유율·갭','해외 동일 셀 성장이 KR을 선행',logit(MOM+FOR)),
   ('v3','v2 + 구조(신규출원인·HHI·우선권유입·공동분류신규조합·제목신규단어)','신규진입·조합신규성이 부상 선행',logit(MOM+FOR+STR)),
   ('v4','v3 + 그룹모멘텀(KSIC·IPC메인그룹 전산업)','산업/기술 단위 추세가 셀로 전파',logit(MOM+FOR+STR+GRP))]
for v,c,h,fp in V:
    R,P=H.rolling(fp,D); H.record(v,c,h,R,P,BASE_AUC,kept='?')

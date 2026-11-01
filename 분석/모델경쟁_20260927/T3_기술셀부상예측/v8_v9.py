from models import *
D=H.data()
# v8: v2 로지스틱 + v5 XGB 순위평균
a=pd.read_parquet('preds/v2.parquet'); b=pd.read_parquet('preds/v5.parquet')
assert (a.cell.values==b.cell.values).all()
def fp8(tr,te):
    return rankavg(logit(MOM+FOR)(tr,te), xgbc(ALL)(tr,te))
R,P=H.rolling(fp8,D); H.record('v8','앙상블: v2 로지스틱 + v5 XGB 순위평균','선형(외삽 안정)·트리(상호작용) 오차가 상보적',R,P,BASE_AUC,kept='?')
R,P=H.rolling(nbfp(ALL),D); H.record('v9','구조모형: XGB Poisson으로 KR 미래 3년 건수 μ 예측 → 음이항 꼬리확률 P(y≥τ), τ=학습원점 상위10% 경계 중앙값','소규모 셀의 계수 잡음(분산)을 명시적으로 모델링하면 breakout 확률 순위가 정확해짐',R,P,BASE_AUC,kept='?')

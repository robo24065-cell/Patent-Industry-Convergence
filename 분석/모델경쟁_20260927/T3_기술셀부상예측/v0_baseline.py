import numpy as np, t3lib as L, harness as H
from sklearn.linear_model import LogisticRegression
def fp(tr,te):
    m=LogisticRegression(C=1e6,max_iter=1000).fit(tr[['kr_mom']].values,tr.yb.values)
    print('coef',m.coef_.ravel())
    return m.predict_proba(te[['kr_mom']].values)[:,1]
R,P=H.rolling(fp)
H.record('v0','고정 기준선: 로지스틱(kr_mom=log((KR t0-2..t0 +1)/(KR t0-5..t0-3 +1)))','과거 3년 KR 모멘텀이 부상을 예측',R,P,baseline_auc=H.summarize(R)['auc'],kept='Y',note='min_total=50 적격, 학습 t0 2005..v-3')
# 참고: 적격 표본 규모
D=H.data(); print(D.groupby('t0').agg(n=('yb','size'),evt=('yb','sum'),kr0=('kr_W',lambda s:(s==0).mean())))

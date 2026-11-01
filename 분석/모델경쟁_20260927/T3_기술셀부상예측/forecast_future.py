"""[2단계 이후 대체됨] 이 스크립트 대신 final_run.py --mode future 를 사용한다 (동결 모형 v30, final_spec.json).
트렌드 산출물(보고서용, 검증 대상 아님) — 홀드아웃 평가 **이후**에만 실행.
t0=2022 기준 2023–2025 KR 부상 예상 기술셀. 학습 레이블: t0 ∈ 2005..2019 (결과창 ≤2022; 출원연도 2023~ 공개지연 절단이라 사용 안 함).
  $env:T3_HOLDOUT_UNLOCK='1'; $env:T3_CACHE='cache_future'; $env:T3_MAXYEAR='2022'; $env:T3_LABEL_MAX='2022'; $env:T3_T0MAX='2022'
  python build_base.py; python build_panel.py; python build_feat2.py; python forecast_future.py
"""
import os
assert os.environ.get('T3_HOLDOUT_UNLOCK')=='1' and os.environ.get('T3_CACHE')=='cache_future'
from models import *
tr=enrich(L.add_labels(pd.concat([L.get_feats(t) for t in range(2005,2020)],ignore_index=True),H.MIN_TOTAL))
te=enrich(L.get_feats(2022)); te=te[te.tot_W>=H.MIN_TOTAL].reset_index(drop=True)
te['y']=np.nan; te['yb']=0
p=best_fp()(tr,te); te['score']=p; te['rank_pct']=te.score.rank(pct=True,ascending=False)
te['pred_breakout_top10']=(te.rank_pct<=0.10).astype(int)
cols=['ksic','mg','kr_W','tot_W','kr_mom','US_mom','CN_mom','JP_mom','EP_mom','score','rank_pct','pred_breakout_top10']
te.sort_values('score',ascending=False)[cols].to_csv('trend_forecast_t0_2022.csv',index=False,encoding='utf-8-sig')
te.groupby('ksic').agg(n_cells=('score','size'),n_pred=('pred_breakout_top10','sum')).assign(rate=lambda d:d.n_pred/d.n_cells).sort_values('rate',ascending=False).to_csv('trend_forecast_ksic_2022.csv',encoding='utf-8-sig')

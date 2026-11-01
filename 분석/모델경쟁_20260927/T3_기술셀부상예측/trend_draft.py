"""트렌드 산출물 초안(검증 구간): t0=2015 기준 v22 예측 상위 셀(KSIC별) + 실제 2016–2018 결과 대조, 국가별 선행 시차 요약표."""
from models import *
import json
D=enrich(H.data()); V=D[D.t0==2015]
P=pd.read_parquet('preds/v22.parquet'); P=P[P.t0==2015].merge(V[['cell','ksic','mg','kr_W','kr_fut','tot_W','kr_mom','US_mom','CN_mom','JP_mom','EP_mom']],on='cell')
P['rank_pct']=P.p.rank(pct=True,ascending=False); P['pred_top10']=(P.rank_pct<=0.10).astype(int)
defs=json.load(open(r'..\..\..\공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json',encoding='utf-8'))
KN={d['code']:d.get('codenmKr','') for d in defs}
def kname(k): return KN.get(k,'')
P['ksic_name']=P.ksic.map(kname)
out=P.sort_values(['ksic','p'],ascending=[True,False]).groupby('ksic').head(5)
out[['ksic','ksic_name','mg','kr_W','tot_W','kr_mom','US_mom','CN_mom','JP_mom','EP_mom','p','rank_pct','pred_top10','kr_fut','y','yb']].to_csv('trend_draft_t0_2015_top5_by_ksic.csv',index=False,encoding='utf-8-sig')
K=P.groupby(['ksic','ksic_name']).agg(n_cells=('p','size'),n_pred=('pred_top10','sum'),n_actual=('yb','sum'),hit=('yb',lambda s: np.nan)).reset_index()
K['hit_in_pred']=P[P.pred_top10==1].groupby('ksic').yb.mean().reindex(K.ksic).values
K['pred_rate']=K.n_pred/K.n_cells; K['actual_rate']=K.n_actual/K.n_cells
K.drop(columns='hit').sort_values('pred_rate',ascending=False).to_csv('trend_draft_t0_2015_by_ksic.csv',index=False,encoding='utf-8-sig')
from scipy.stats import spearmanr
print('KSIC 수준: 예측 부상비율 vs 실제 부상비율 Spearman', spearmanr(K.pred_rate,K.actual_rate))
print('상위10% 예측 셀 적중률', P[P.pred_top10==1].yb.mean(), '예측 셀 수', P.pred_top10.sum())
LL=pd.read_csv('leadlag_summary.csv')
piv=LL.pivot_table(index=['target','cc'],columns='lag',values=['z','delta_auc']).round(3)
piv.to_csv('leadlag_pivot.csv',encoding='utf-8-sig'); print(piv)
print(out.head(12)[['ksic','ksic_name','mg','kr_W','p','kr_fut','yb']].to_string())

# 보조타깃A(KR 과거3년 ≥5건) 기준 초안 — 실질적 부상
PA=pd.read_parquet('preds/v25.parquet'); PA=PA[PA.t0==2015].merge(V[['cell','ksic','mg','kr_W','kr_fut','tot_W','US_mom','CN_mom','JP_mom','EP_mom']],on='cell')
PA['rank_pct']=PA.p.rank(pct=True,ascending=False); PA['pred_top10']=(PA.rank_pct<=0.10).astype(int); PA['ksic_name']=PA.ksic.map(kname)
PA.sort_values('p',ascending=False).head(60)[['ksic','ksic_name','mg','kr_W','tot_W','US_mom','CN_mom','JP_mom','EP_mom','p','kr_fut','y','yb']].to_csv('trend_draft_t0_2015_targetA_top60.csv',index=False,encoding='utf-8-sig')
print('보조A 상위10% 적중률',PA[PA.pred_top10==1].yb.mean(),'절대증가 중앙값',(PA[PA.pred_top10==1].kr_fut-PA[PA.pred_top10==1].kr_W).median())
print(PA.sort_values('p',ascending=False).head(10)[['ksic_name','mg','kr_W','kr_fut','yb']].to_string())

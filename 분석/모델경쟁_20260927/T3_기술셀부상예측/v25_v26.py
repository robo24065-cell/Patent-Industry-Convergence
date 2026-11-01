from models import *
D=enrich(H.data())
def base_fp(tr,te):
    m=LogisticRegression(C=1e6,max_iter=1000).fit(tr[['kr_mom']].values,tr.yb.values); return m.predict_proba(te[['kr_mom']].values)[:,1]
# v25: 의미 있는 부상 — KR 과거 3년 ≥5건 셀만
DA=relabel(D, D.kr_W>=5)
R0,_=H.rolling(base_fp,DA); b=H.summarize(R0)['auc']; print('alt-A baseline',H.summarize(R0))
R,P=H.rolling(best_fp(),DA); H.record('v25','보조타깃A: 적격에 KR 과거3년 ≥5건 추가(소규모 0→3 류 제외) + v22 파이프라인. 기준선도 같은 표본에서 재계산',
  '주 타깃 상위 선정 셀의 절대 증가 중앙값이 1건뿐 → 실질적 부상만 남겨도 예측력이 유지되는지',R,P,b,kept='N',note=f'보조 타깃(주 지표와 직접 비교 불가). 표본/사건: '+'/'.join(f"{n}/{e}" for n,e in zip(R.n,R.n_evt)))
# v26: 보조타깃B — 5개국 합계 성장
P_=L.load(); tot=P_['A']['n'].sum(1); idx=np.searchsorted(P_['cells'],D.cell.values)
fut=np.zeros(len(D))
for t,g in D.groupby('t0'): fut[g.index]=L.wsum(tot,t,1,3)[idx[g.index]]
D['tot_fut']=fut; D['y_tot']=np.log((fut+1)/(D.tot_W+1)); D['tie_tot']=fut-D.tot_W
DB=relabel(D,ycol='y_tot',tiecol='tie_tot'); DB['y']=DB['y_tot']
def base_fpB(tr,te):
    m=LogisticRegression(C=1e6,max_iter=1000).fit(tr[['tot_mom']].values,tr.yb.values); return m.predict_proba(te[['tot_mom']].values)[:,1]
R0,_=H.rolling(base_fpB,DB); b=H.summarize(R0)['auc']; print('alt-B baseline',H.summarize(R0))
R,P=H.rolling(best_fp(extra_nb=False),DB); H.record('v26','보조타깃B: 5개국 합계 성장 상위10% (기준선=5개국 합계 모멘텀 로지스틱), v22 파이프라인(NB 성분 제외)',
  '전세계 기술셀 부상도 같은 피처로 예측 가능한지',R,P,b,kept='N',note='보조 타깃. CN 비중 커서 사실상 CN 주도 성장')

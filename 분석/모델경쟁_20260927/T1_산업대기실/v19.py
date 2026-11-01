# v19: 진단3 반영 — 외국법인/국가기관 플래그(학습 레이블 등록 절단과 상호작용해 편향)는 빼고, 마지막 출원 경과·최근1년 출원만 추가
from data8 import *
t0 = time.time(); tr, te = load19(); base = json.load(open(T1/'baseline_v0.json'))['fixed']
s, sx = run_std(tr, te, C19); met = metrics(te, s); mx = metrics(te, sx)
print('v19 xgb only', fmt(mx)); print('v19', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v19_scores.parquet', index=False)
lb_add('v19', 'v14 + 마지막 출원 경과·최근1년 출원 수만(외국/국가기관 플래그 제외)', '진단3: 외국법인 플래그는 학습 레이블 등록절단(PCT 지연)과 얽혀 반대로 학습됨. 활동 최근성 두 개만이 순수 이득', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}; 진단3 1시드: foreign+gov 추가시 auc_new 0.8513→0.8261")

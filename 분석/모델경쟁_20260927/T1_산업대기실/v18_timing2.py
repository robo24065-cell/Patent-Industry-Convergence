# v18: v17 진단 — gap_first·act_years·age는 학습 원점(2002~04, 이력 ≤5년)과 평가 원점(2009, 이력 ≤10년)에서 분포 범위가 달라 트리가 외삽 실패.
#      이력 길이에 불변인 형태만 남긴다: gap_last를 0/1/2+로 절단, 최근 1·2년 출원 비중(최근/전체), 외국법인·국가기관.
from data8 import *
t0 = time.time(); cuts = (2002, 2003, 2004)
tr, te = load14(cuts)
xtr = pd.concat([tim_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = tim_panel(2009, 'None')
for d, x in ((tr, xtr), (te, xte)):
    d['gap_last2'] = np.minimum(x.gap_last.values, 2); d['foreign'] = x.foreign.values; d['gov'] = x.gov.values
    tot = np.expm1(d.lsize.values)
    d['sh_last1'] = np.expm1(x.n_last1.values) / np.maximum(tot, 1); d['sh_last2'] = np.expm1(x.n_last2.values) / np.maximum(tot, 1)
    d['n_last1'] = x.n_last1.values
T2 = ['gap_last2', 'foreign', 'gov', 'sh_last1', 'sh_last2', 'n_last1']
base = json.load(open(T1/'baseline_v0.json'))['fixed']
P = json.load(open(T1/'v9_cv.json'))['g4']['params']; cols = C14 + T2
sx = np.mean([predict(fit_xgb(tr, cols, seed=sd, **P), te, cols) for sd in range(3)], 0); sl = logit(tr, te, cols)
s = 0.9 * rank(sx) + 0.1 * rank(sl); met = metrics(te, s); mx = metrics(te, sx)
print('v18 xgb only', fmt(mx)); print('v18', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v18_scores.parquet', index=False)
lb_add('v18', 'v17 수정: 이력 길이 의존 피처(첫출원 경과·활동연수·설립경과) 제거, 마지막 출원 경과 0/1/2+ 절단, 최근 1·2년 출원 비중으로 대체', 'v17 AUC 하락은 학습(이력≤5년)·평가(≤10년) 원점 간 분포 이동 때문 → 이력 길이 불변 형태면 PR 이득은 살리고 AUC 손실은 없앤다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")

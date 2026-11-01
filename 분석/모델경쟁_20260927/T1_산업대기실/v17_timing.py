# v17: 법인 활동 시점(타이밍) 피처 — 진단2: 신규진입자 오차의 병목은 '어디로'(법인내 AUC 0.868)가 아니라 '언제/누가'(법인 선택 AUC 0.544)
from data8 import *
t0 = time.time(); cuts = (2002, 2003, 2004)
tr, te = load14(cuts)
xtr = pd.concat([tim_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = tim_panel(2009, 'None')
for c in TIMF: tr[c] = xtr[c].values; te[c] = xte[c].values
base = json.load(open(T1/'baseline_v0.json'))['fixed']
P = json.load(open(T1/'v9_cv.json'))['g4']['params']; cols = C14 + TIMF
sx = np.mean([predict(fit_xgb(tr, cols, seed=sd, **P), te, cols) for sd in range(3)], 0); sl = logit(tr, te, cols)
s = 0.9 * rank(sx) + 0.1 * rank(sl); met = metrics(te, s); mx = metrics(te, sx)
print('v17 xgb only', fmt(mx)); print('v17', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v17_scores.parquet', index=False)
lb_add('v17', 'v14 + 법인 활동 시점 피처(마지막·첫 출원 경과, 최근1·2년 출원, 최근3년 범위안, 활동연수, 설립경과, 외국법인·국가기관)', '진단2: 신규진입자는 산업 선택(법인내 AUC 0.868)보다 진입 시점(법인 선택 AUC 0.544)이 병목 → 활동 시점 정보가 필요', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")

# v14: 기존 범위안 특허 제목의 2차 산업 확률(교차적합) — 기존진입자의 잠재 다각화 신호
from data8 import *
t0 = time.time(); cuts = (2002, 2003, 2004)
tr, te = load13(cuts)
xtr = pd.concat([ins_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = ins_panel(2009, 'None')
for c in INSF: tr[c] = xtr[c].values; te[c] = xte[c].values
base = json.load(open(T1/'baseline_v0.json'))['fixed']
P = json.load(open(T1/'v9_cv.json'))['g4']['params']; cols = C13 + INSF
sx = np.mean([predict(fit_xgb(tr, cols, seed=sd, **P), te, cols) for sd in range(3)], 0); sl = logit(tr, te, cols)
s = 0.9 * rank(sx) + 0.1 * rank(sl); met = metrics(te, s); mx = metrics(te, sx)
print('v14 xgb only', fmt(mx)); print('v14', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v14_scores.parquet', index=False)
lb_add('v14', 'v13 + 기존 범위안 특허 제목의 2차 산업 확률(2-fold 교차적합: 평균·최대·p>0.2 건수·로그합)', '범위안 특허도 제목상 다른 산업 성분을 품고 있고, 그 성분이 다음 진입 산업을 가리킨다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")

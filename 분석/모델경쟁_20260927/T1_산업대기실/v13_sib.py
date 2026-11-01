# v13: KSIC 계층 형제 산업 피처(4/3/2자리 형제 산업 보유·형제 산업 텍스트 노출)
from data8 import *
t0 = time.time(); cuts = (2002, 2003, 2004)
tr, te = load8(cuts); tr = add_ep(tr); te = add_ep(te)
xtr = pd.concat([sib_panel(c, CAP_TR) for c in cuts], ignore_index=True); xte = sib_panel(2009, 'None')
for c in SIBF: tr[c] = xtr[c].values; te[c] = xte[c].values
base = json.load(open(T1/'baseline_v0.json'))['fixed']
P = json.load(open(T1/'v9_cv.json'))['g4']['params']; cols = C8 + EPF + SIBF
sx = np.mean([predict(fit_xgb(tr, cols, seed=sd, **P), te, cols) for sd in range(3)], 0); sl = logit(tr, te, cols)
s = 0.9 * rank(sx) + 0.1 * rank(sl); met = metrics(te, s); mx = metrics(te, sx)
print('v13 xgb only', fmt(mx)); print('v13', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v13_scores.parquet', index=False)
lb_add('v13', 'v11 + KSIC 계층 형제 산업(같은 4/3/2자리 산업 보유 수, 형제 산업 텍스트 노출 합)', '공식 분류 계층상 형제 산업 경험은 동시보유 근접도가 놓치는 기술 인접성을 준다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")

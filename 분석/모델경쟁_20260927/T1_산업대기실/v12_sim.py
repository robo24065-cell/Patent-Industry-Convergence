# v12: 산업 프런티어/최근 진입자 프로파일 유사도(중심벡터 코사인) 추가
from data8 import *
t0 = time.time()
cuts = (2002, 2003, 2004)
tr, te = load8(cuts); tr = add_ep(tr); te = add_ep(te)
xs = add_sim(None, cuts); xtr = pd.concat(xs[:-1], ignore_index=True); xte = xs[-1]
assert len(xtr) == len(tr) and len(xte) == len(te)
for c in SIMF: tr[c] = xtr[c].values; te[c] = xte[c].values
print('sim feats ready', round(time.time() - t0), flush=True)
base = json.load(open(T1/'baseline_v0.json'))['fixed']
P = json.load(open(T1/'v9_cv.json'))['g4']['params']; cols = C8 + EPF + SIMF
sx = np.mean([predict(fit_xgb(tr, cols, seed=sd, **P), te, cols) for sd in range(3)], 0); sl = logit(tr, te, cols)
s = 0.9 * rank(sx) + 0.1 * rank(sl); met = metrics(te, s); mx = metrics(te, sx)
print('v12 xgb only', fmt(mx)); print('v12', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v12_scores.parquet', index=False)
lb_add('v12', '+산업 프런티어 유사도(최근5년 범위안 중심벡터)·최근 진입자 첫특허 프로파일 유사도(평균·최대·상위3·법인내 상대값)', '분류기(누적 라벨)보다 최근 진입자의 모습과 닮은 기업이 다음 진입자다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}")

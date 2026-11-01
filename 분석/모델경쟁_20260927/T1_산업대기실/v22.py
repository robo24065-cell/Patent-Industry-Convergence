# v22: 학습 원점 2001 추가(2001~2004 스태킹) + IPW
import sys
from data8 import *
t0 = time.time(); cuts = (2001, 2002, 2003, 2004)
tr, te = load19(cuts); base = json.load(open(T1/'baseline_v0.json'))['fixed']
w = ipw_weights(tr); print('train rows', len(tr), 'pos', int(tr.y.sum()))
s, sx = run_std(tr, te, C19, w=w); met = metrics(te, s); mx = metrics(te, sx)
print('v22 xgb only', fmt(mx)); print('v22', fmt(met), round(time.time() - t0))
te[['firm', 'k', 'y', 'newc']].assign(s=s, sx=sx).to_parquet(CACHE/'v22_scores.parquet', index=False)
lb_add('v22', 'v20 + 학습 원점 2001 추가(2001~2004 스태킹)', 'IPW로 절단을 보정했으니 이른 원점을 더 쌓아 표본을 늘릴 수 있다', met, base, '?', f"xgb단독 auc_all={mx['auc_all']:.4f} auc_new={mx['auc_new']:.4f}; train rows={len(tr)}")

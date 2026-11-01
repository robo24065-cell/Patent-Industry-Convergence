# 내부 시간분할 튜닝: 학습 L2006-2008 → 내부검증 L2011 (레이블 [2011,2014) 은 2014-01-01 전 관측 완료). 검증(2014/15)은 보지 않음.
# 사용: python tune.py <featset> [grid-part]
import sys, time, itertools, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import run
from common import *
fs = sys.argv[1]; part = int(sys.argv[2]) if len(sys.argv) > 2 else 0
F = {'full': run.FULL, 'full_unit': run.FULL + run.UNIT}[fs]
tr = run.load(range(2006, 2009)); iv = run.load([2011])
grid = list(itertools.product([3, 4, 6, 8], [50, 300], [0.5, 0.8], [1.0, 10.0]))
grid = grid[part * 6:(part + 1) * 6] if part < 10 else {10: [(4, 50, 0.5, 10.0), (6, 50, 0.5, 10.0), (6, 10, 0.5, 10.0)], 11: [(8, 50, 0.5, 10.0), (6, 20, 0.8, 1.0), (2, 20, 0.5, 10.0)], 12: [(6, 2, 0.5, 10.0), (8, 5, 0.5, 10.0), (6, 5, 0.5, 1.0)], 13: [(10, 5, 0.5, 10.0), (8, 2, 0.5, 10.0), (4, 5, 0.5, 10.0)]}[part]
out = []
for depth, mcw, col, lam in grid:
    t = time.time()
    m = xgb.XGBClassifier(n_estimators=1500, max_depth=depth, learning_rate=0.03, subsample=0.8, colsample_bytree=col, min_child_weight=mcw,
                          reg_lambda=lam, nthread=2, tree_method='hist', eval_metric='auc', early_stopping_rounds=100, random_state=0)
    m.fit(tr[F], tr.y_MAIN, eval_set=[(iv[F], iv.y_MAIN)], verbose=False)
    p = m.predict_proba(iv[F])[:, 1]
    r = dict(depth=depth, mcw=mcw, col=col, lam=lam, best_it=int(m.best_iteration), auc=roc_auc_score(iv.y_MAIN, p), pr=average_precision_score(iv.y_MAIN, p))
    out.append(r); print(r, round(time.time() - t, 1), flush=True)
pd.DataFrame(out).to_csv(f'tune_{fs}_{part}.csv', index=False)

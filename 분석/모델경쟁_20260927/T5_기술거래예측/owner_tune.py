# 권리자 모형 내부 시간분할 튜닝: 학습 L2006-2008 → 내부검증 L2011, 특허 단위 AUC(owner_hat 부여)로 평가. 검증(2014/15) 미사용.
import sys, itertools, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import run
from owner_model_lib import owner_table, feats
from common import *
part = int(sys.argv[1])
tr = run.load(range(2006, 2009)); iv = run.load([2011]); tr['ytr'] = tr.y_MAIN; iv['ytr'] = iv.y_MAIN
ot = owner_table(tr); oi = owner_table(iv)
F = feats('A')
grid = list(itertools.product([3, 5, 7], [1, 5, 20], [0.5, 0.8]))[part * 6:(part + 1) * 6]
for depth, mcw, col in grid:
    t = time.time()
    m = xgb.XGBRegressor(objective='reg:logistic', n_estimators=2000, max_depth=depth, learning_rate=0.03, subsample=0.8, colsample_bytree=col,
                         min_child_weight=mcw, reg_lambda=5.0, nthread=2, tree_method='hist', random_state=0, early_stopping_rounds=100, eval_metric='logloss')
    m.fit(ot[F], ot.rate, sample_weight=np.sqrt(ot.n), eval_set=[(oi[F], oi.rate)], sample_weight_eval_set=[np.sqrt(oi.n)], verbose=False)
    oi['h'] = m.predict(oi[F])
    d = iv[['L', 'cur', 'y_MAIN']].merge(oi[['L', 'cur', 'h']], on=['L', 'cur'])
    print(dict(depth=depth, mcw=mcw, col=col, best_it=m.best_iteration, auc=round(roc_auc_score(d.y_MAIN, d.h), 4)), round(time.time() - t, 1), flush=True)

# v9: XGB 하이퍼파라미터 — 학습 패널 내부 GroupKFold(법인) CV로만 선택(검증 구간은 선택에 쓰지 않음)
import sys
from data8 import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
t0 = time.time()
tr, te = load8()
grid = {'g0': dict(), 'g1': dict(max_depth=3, n_estimators=800, learning_rate=0.03), 'g2': dict(max_depth=4, n_estimators=800, learning_rate=0.03, min_child_weight=20),
        'g3': dict(max_depth=6, n_estimators=600, learning_rate=0.03, min_child_weight=20, colsample_bytree=0.5), 'g4': dict(max_depth=4, n_estimators=1200, learning_rate=0.02, min_child_weight=10, reg_lambda=5, colsample_bytree=0.6)}
which = sys.argv[1:] or list(grid)
res = json.load(open(T1/'v9_cv.json')) if (T1/'v9_cv.json').exists() else {}
gkf = GroupKFold(n_splits=3)
for g in which:
    aucs = []; aucn = []
    for a, b in gkf.split(tr, groups=tr.firm):
        m = fit_xgb(tr.iloc[a], C8, **grid[g]); s = predict(m, tr.iloc[b], C8); yb = tr.y.values[b]; nb = tr.newc.values[b]
        aucs.append(roc_auc_score(yb, s)); aucn.append(roc_auc_score(yb[nb], s[nb]))
    res[g] = dict(cv_auc=float(np.mean(aucs)), cv_auc_new=float(np.mean(aucn)), params=grid[g]); print(g, res[g], round(time.time() - t0), flush=True)
    json.dump(res, open(T1/'v9_cv.json', 'w'), indent=1)

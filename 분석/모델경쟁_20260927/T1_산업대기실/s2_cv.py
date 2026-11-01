# 엄격 정보집합 하이퍼파라미터 내부 CV(학습 패널만, 법인 GroupKFold 3) — 검증 c=2009 레이블은 보지 않는다.
# 실행: T1_STRICT=1 python s2_cv.py s2p0 s2p1 ...   (fold별 캐시, 9분 넘으면 재실행)
import os, sys; os.environ['T1_STRICT'] = '1'
from s2lib import *
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
GRID = {
    's2p0': dict(max_depth=4, n_estimators=1200, learning_rate=0.02, min_child_weight=10, reg_lambda=5, colsample_bytree=0.6),   # = g4(현행)
    's2p1': dict(max_depth=3, n_estimators=1500, learning_rate=0.03, min_child_weight=20, reg_lambda=5, colsample_bytree=0.5),
    's2p2': dict(max_depth=5, n_estimators=1000, learning_rate=0.02, min_child_weight=30, reg_lambda=10, colsample_bytree=0.5),
    's2p3': dict(max_depth=4, n_estimators=1200, learning_rate=0.02, min_child_weight=50, reg_lambda=5, colsample_bytree=0.4),
    's2p4': dict(max_depth=6, n_estimators=800, learning_rate=0.02, min_child_weight=50, reg_lambda=10, colsample_bytree=0.4),
}
t0 = time.time(); tr, _ = load_s2()
COLS = [c for c in C32 if c not in DART]
w = ipw_weights(tr); y = tr.y.values; nc = tr.newc.values
folds = list(GroupKFold(3).split(tr, y, tr.firm.values)); X = tr[COLS].values.astype(np.float32)
OUT = T1/'s2_cv.json'; res = json.load(open(OUT)) if OUT.exists() else {}
CV = T1/'logs'/'cv'; CV.mkdir(exist_ok=True)
for g in sys.argv[1:]:
    oof = np.zeros(len(tr)); ok = True
    for i, (a, b) in enumerate(folds):
        fp = CV/f'{g}_{i}.npy'
        if fp.exists(): oof[b] = np.load(fp); continue
        if time.time() - t0 > 380: ok = False; print('재실행 필요', g, i); break
        p = dict(subsample=0.8, tree_method='hist', nthread=2, random_state=0); p.update(GRID[g])
        m = xgb.XGBClassifier(**p).fit(X[a], y[a], sample_weight=w[a]); oof[b] = m.predict_proba(X[b])[:, 1]; np.save(fp, oof[b])
        print(g, 'fold', i, round(time.time() - t0), flush=True)
    if not ok: sys.exit(3)
    res[g] = dict(cv_auc=float(roc_auc_score(y, oof)), cv_auc_new=float(roc_auc_score(y[nc], oof[nc])), params=GRID[g])
    print(g, res[g], flush=True); json.dump(res, open(OUT, 'w'), indent=1)

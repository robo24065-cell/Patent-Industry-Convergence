# 최종 후보 v31 동결 모형 학습·저장(엄격 정보집합). 재실행하면 없는 시드만 학습.
# 실행: T1_STRICT=1 T1_VIS=rgst python final_model.py
import os; assert os.environ.get('T1_STRICT') == '1' and os.environ.get('T1_VIS', 'rgst') == 'rgst'
from data8 import *
import xgboost as xgb, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
CUTS = (2001, 2002, 2003, 2004)
def features(d, cuts, te=None):
    pass
def load_final(cuts=CUTS, eval_cut=2009):
    tr, te = load19(cuts)
    tr = add_pa(tr); te = add_pa(te)
    tr, te = add_rr(tr, te, cuts); tr, te = add_now(tr, te, cuts)
    return tr, te
COLS = C19 + PAF + RRF + NOWF
if __name__ == '__main__':
    t0 = time.time(); tr, _ = load_final(); w = ipw_weights(tr)
    P = json.load(open(T1/'v9_cv.json'))['g4']['params']
    for sd in range(3):
        fp = T1/f'models/v31_xgb_seed{sd}.json'
        if fp.exists(): continue
        if time.time() - t0 > 400: print('다시 실행 필요'); break
        p = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method='hist', nthread=2, random_state=sd); p.update(P)
        m = xgb.XGBClassifier(**p); m.fit(tr[COLS].values.astype(np.float32), tr.y.values, sample_weight=w); m.save_model(fp); print('saved', fp.name, round(time.time() - t0), flush=True)
    fl = T1/'models/v31_logit.pkl'
    if not fl.exists():
        cl = [c for c in COLS if c != 'ind2']; X = tr[cl].fillna(0).values; sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(X), tr.y); pickle.dump(dict(sc=sc, lr=lr, cols=cl), open(fl, 'wb')); print('saved logit')
    json.dump(dict(cols=COLS, cuts=CUTS, params=P, blend='0.9*rank(mean xgb 3 seeds)+0.1*rank(logit)', ipw='data8.ipw_weights(cap=20091231, cohort<=2002, wmax=5)'), open(T1/'models/v31_spec.json', 'w'), indent=1, ensure_ascii=False)

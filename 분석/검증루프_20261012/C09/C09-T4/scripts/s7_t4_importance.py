# C09-T4 s7: T4 최종창(F22) 지연 피처 기여 — (i) cls bag0 총 gain 순위, (ii) 지연 5개 열 묶음 순열(특허 단위, 시드 0) 앙상블 AUC 하락.
#  저장 모형(cache/t4pred/F22/<rule>/)으로 예측. 위험모형 결정행은 같은 특허의 순열값을 받는다.
#  python s7_t4_importance.py <rule>
import os, sys, json, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
from s2_t4_feats import LATE, fdir
from s3_t4_train import SPEC, CAT, read, pdir, _prep_cat, combine_haz, LogPrep
import __main__; __main__.LogPrep = LogPrep   # s3 를 __main__ 으로 실행해 저장한 피클 복원용
from sklearn.metrics import roc_auc_score
import xgboost as xgb
o = 'F22'; rule = sys.argv[1]
t = time.time()
WS = SPEC['ensemble']['weights']
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
allc = set()
for k in WS: allc |= set(SPEC['components'][k]['num'])
e = read(o, 'ev', list(dict.fromkeys([c for c in allc] + CAT + ['pid', 'y'])), rule)
eh = read(o, 'ev_haz', list(dict.fromkeys(SPEC['components']['haz']['num'] + CAT + ['pid', 'row'])), rule)
y = e.y.values; m = ~np.isnan(y)
pb = np.load(os.path.join(fdir(o), 'pred', 'base_46526c56d8.npy'))


def predict(e, eh):
    ps = {}
    for comp in ['cls', 'haz']:
        X0 = e if comp == 'cls' else eh
        zs = []
        for b in range(3):
            bst = xgb.Booster(); bst.load_model(os.path.join(pdir(o, rule), f'{comp}_b{b}.json'))
            mm = pickle.load(open(os.path.join(pdir(o, rule), f'{comp}_b{b}.pkl'), 'rb'))
            X = X0[mm['num'] + CAT].copy()
            for c in mm['num']:
                if X[c].dtype != np.float32: X[c] = X[c].astype('float32')
            X, _ = _prep_cat(X, mm['cmap'])
            p = np.clip(bst.predict(xgb.DMatrix(X, enable_categorical=True)), 1e-6, 1 - 1e-6)
            zs.append(np.log(p / (1 - p)))
        z = np.mean(zs, 0)
        ps[comp] = 1 / (1 + np.exp(-z)) if comp == 'cls' else combine_haz(o, z)
    cfgl = SPEC['components']['logit']; mm = pickle.load(open(os.path.join(pdir(o, rule), 'logit.pkl'), 'rb'))
    ps['logit'] = mm['m'].predict_proba(mm['prep'].transform(e))[:, 1]
    cfgh = SPEC['components']['hgb']; mm = pickle.load(open(os.path.join(pdir(o, rule), 'hgb.pkl'), 'rb'))
    ce = mm['oe'].transform(e[CAT].astype(str)).astype('float32'); ce[ce < 0] = np.nan
    Xe = np.c_[e[cfgh['num']].to_numpy(dtype='float32', na_value=np.nan), ce]; Xe[:, mm['allnan']] = 0
    ps['hgb'] = mm['m'].predict_proba(Xe)[:, 1]
    zz = sum(w * lg(ps[k]) for k, w in WS.items()) / sum(WS.values())
    return 1 / (1 + np.exp(-zz)), ps


p0, ps0 = predict(e, eh)
a0 = roc_auc_score(y[m], p0[m])
# 저장 예측과 대조
chk = {}
for k in ['logit', 'hgb']:
    chk[k] = float(np.abs(ps0[k] - np.load(os.path.join(pdir(o, rule), f'{k}.npy'))).max())
rng = np.random.default_rng(0); perm = rng.permutation(len(e))
ep = e.copy(); ehp = eh.copy()
L0 = e[LATE].to_numpy(dtype='float32', na_value=np.nan)
ep[LATE] = L0[perm]
ehp[LATE] = L0[perm][eh.row.values]
p1, ps1 = predict(ep, ehp)
a1 = roc_auc_score(y[m], p1[m])
comp_drop = {k: float(roc_auc_score(y[m], ps0[k][m]) - roc_auc_score(y[m], ps1[k][m])) for k in ps0}
g = json.load(open(os.path.join(pdir(o, rule), 'cls_b0_gain.json')))
s = pd.Series(g).sort_values(ascending=False)
rk = s.rank(ascending=False, method='min')
out = dict(rule=rule, auc=float(a0), auc_perm=float(a1), perm_auc_drop=float(a0 - a1), comp_perm_drop=comp_drop,
           skill=float((a0 - roc_auc_score(y[m], pb[m])) / (1 - roc_auc_score(y[m], pb[m]))),
           gain_rank={c: (int(rk[c]) if c in rk else None) for c in LATE}, n_feat=int(len(s)), top10=s.head(10).index.tolist(),
           saved_pred_check_maxabs=chk)
json.dump(out, open(os.path.join(C, f't4_importance_{rule}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False), f'{time.time() - t:.0f}s')

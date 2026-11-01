# C13-T4 s5: X1' P0(b) — 실제 모형 예측 차이의 이원 군집 부트스트랩 SE와 검정력(PREREG 변경 2).
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from lib13 import *

BASE = ['b_n3', 'b_n1', 'b_gr', 'b_npart', 'b_new3', 'b_new1', 'b_cosh', 'b_nsc', 'b_iscn']
SX = ['x_nk', 'x_ent', 'x_msh', 'x_xpart', 'x_xpart_na', 'x_xnew3', 'x_indr', 'x_indn', 'x_imp', 'x_imp_na', 'x_modal_na']
TR = list(range(2016, 2023)); EV = 2023
LOCK_EXP = 1981.5073339898618
TARGET = 0.002


def load(sys_):
    return pd.concat([pd.read_parquet(f'{C}/u_{sys_}_{t}.parquet').assign(t0=t) for t in TR + [EV]], ignore_index=True)


P = load('prov')
F = {'prov': P}
for s in ['free', 'hsc']:
    F[s] = load(s)
    assert (F[s].u.values == P.u.values).all() and (F[s].t0.values == P.t0.values).all()
y = P.y.values.astype(int)
tr = P.t0.isin(TR).values; ev = (P.t0 == EV).values


def fit_pred(X):
    m = xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=2, random_state=SEED, eval_metric='logloss')
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[ev])[:, 1]


pred = {}
pred['base'] = fit_pred(P[BASE].values)
for s in ['prov', 'free', 'hsc']:
    pred[s] = fit_pred(np.c_[P[BASE].values, F[s][SX].values])
ye = y[ev]
auc = {k: roc_auc_score(ye, v) for k, v in pred.items()}
pub = 'free' if auc['free'] >= auc['hsc'] else 'hsc'
out('AUC', auc, 'pub=', pub)
pd.DataFrame({'u': P.u[ev].values, 'y': ye, 'modal_prov': P.modal[ev].values, **{f'p_{k}': v for k, v in pred.items()}}).to_parquet(f'{C}/s5_pred_2023.parquet', index=False)

# 이원 군집 부트스트랩
rng = np.random.default_rng(SEED)
units = np.arange(ev.sum())
ks = P.modal[ev].fillna('NA').values
uk, kinv = np.unique(ks, return_inverse=True)
B_ = 500
d_two, d_firm, d_two_base = [], [], []
for b in range(B_):
    wf = rng.poisson(1.0, len(units)).astype(float)
    wk = rng.poisson(1.0, len(uk)).astype(float)[kinv]
    for w, lst in ((wf * wk, d_two), (wf, d_firm)):
        if w[ye == 1].sum() == 0 or w[ye == 0].sum() == 0:
            continue
        lst.append(roc_auc_score(ye, pred['prov'], sample_weight=w) - roc_auc_score(ye, pred[pub], sample_weight=w))
    w = wf * wk
    d_two_base.append(roc_auc_score(ye, pred['prov'], sample_weight=w) - roc_auc_score(ye, pred['base'], sample_weight=w))
se_two = float(np.std(d_two, ddof=1)); se_firm = float(np.std(d_firm, ddof=1))
npos = int(ye.sum())
scale = np.sqrt(npos / LOCK_EXP)
res = {'eval_t0': EV, 'n_eval': int(ev.sum()), 'pos_eval': npos, 'auc': auc, 'pub_arm': pub,
       'dAUC_prov_pub': auc['prov'] - auc[pub], 'dAUC_prov_base': auc['prov'] - auc['base'],
       'SE_dev_two': se_two, 'SE_dev_firm': se_firm, 'ci_two': [float(np.percentile(d_two, 2.5)), float(np.percentile(d_two, 97.5))],
       'ci_firm': [float(np.percentile(d_firm, 2.5)), float(np.percentile(d_firm, 97.5))],
       'ci_two_prov_base': [float(np.percentile(d_two_base, 2.5)), float(np.percentile(d_two_base, 97.5))],
       'scale_sqrt_pos': float(scale), 'SE_lock_two': se_two * scale, 'SE_lock_firm': se_firm * scale,
       'target_dAUC': TARGET}
res['power_two'] = float(norm.cdf(TARGET / res['SE_lock_two'] - 1.96))
res['power_firm'] = float(norm.cdf(TARGET / res['SE_lock_firm'] - 1.96))
res['P0b_pass'] = bool(res['power_two'] >= 0.80)
res['n_boot'] = len(d_two)
json.dump(res, open(f'{C}/s5_p0b.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out(json.dumps(res, ensure_ascii=False, indent=1))

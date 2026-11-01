# C05-T4 모형 공용 함수(개발 전진 평가와 동결 학습이 같은 코드를 쓴다).
import os
import numpy as np, pandas as pd
from libC import C, DEV, LOCK
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

M_TE = 1000
BASE0 = ['lE0', 'lnw', 'age', 'demp', 'newkey', 'noprev', 'r_new', 'r_loss']
TE = ['te_ind2', 'te_ind4', 'te_ind6', 'te_sido', 'te_sgg']
M0 = BASE0 + TE
PEN = ['pen_ind6', 'pen_sgg', 'flow_ind6']
FEATS = {'M0': M0,
         'M1': M0 + PEN + ['br_gKR_prov', 'br_gF4_prov'],
         'M1_pub_TXT': M0 + PEN + ['br_gKR_TXT', 'br_gF4_TXT'],
         'M1_pub_IPC': M0 + PEN + ['br_gKR_IPC', 'br_gF4_IPC']}
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  tree_method='hist', n_jobs=2, random_state=0, eval_metric='logloss')
LR_PARAMS = dict(C=1.0, solver='lbfgs', max_iter=2000)


def load_table(origins):
    u = pd.read_parquet(os.path.join(C, 'units.parquet'))
    u = u[u.s0.isin(origins)].reset_index(drop=True)
    br = pd.read_parquet(os.path.join(C, 'bridge.parquet'))
    u = u.merge(br, on=['key', 's0'], how='left', validate='one_to_one')
    assert u.filter(like='br_').notna().all().all()
    u['lE0'] = np.log1p(u.E); u['lnw'] = np.log1p(u.n_wkpl)
    u['r_new'] = u.n_new / u.E.clip(lower=1); u['r_loss'] = u.n_loss / u.E.clip(lower=1)
    return u


def _enc(fit, app, chain, m=M_TE):
    g = fit.y.mean()
    prev = pd.Series(g, index=app.index, dtype=float); out = {}
    for lv in chain:
        st = fit.groupby(lv).y.agg(['sum', 'count'])
        s = app[lv].map(st['sum']).fillna(0.0).astype(float); n = app[lv].map(st['count']).fillna(0.0).astype(float)
        cur = (s + m * prev) / (n + m)
        out[lv] = np.log(cur); prev = cur
    return out


def add_te(train, test):
    """목표 인코딩: 학습 행은 같은 학습 집합의 다른 원점으로(원점 밖), 평가 행은 학습 원점 전부로."""
    assert (train.y >= 0).all()
    train = train.copy(); test = test.copy()
    for c in TE:
        train[c] = np.nan
    for s in train.s0.unique():
        m = train.s0 == s
        fit = train[~m]
        if len(fit) == 0:
            fit = train[m]  # (사용 안 함: 학습 원점은 항상 ≥3개)
        e1 = _enc(fit, train[m], ['ind2', 'ind4', 'ind6']); e2 = _enc(fit, train[m], ['sido', 'sgg'])
        train.loc[m, 'te_ind2'] = e1['ind2']; train.loc[m, 'te_ind4'] = e1['ind4']; train.loc[m, 'te_ind6'] = e1['ind6']
        train.loc[m, 'te_sido'] = e2['sido']; train.loc[m, 'te_sgg'] = e2['sgg']
    e1 = _enc(train, test, ['ind2', 'ind4', 'ind6']); e2 = _enc(train, test, ['sido', 'sgg'])
    test['te_ind2'] = e1['ind2']; test['te_ind4'] = e1['ind4']; test['te_ind6'] = e1['ind6']
    test['te_sido'] = e2['sido']; test['te_sgg'] = e2['sgg']
    return train, test


def _lr_matrix(df, cols, med):
    X = df[cols].copy()
    for c in ['demp', 'newkey']:
        if c in X:
            X[c] = X[c].fillna(0.0)
    return X.fillna(med).values.astype(float)


def fit_predict(train, test, model):
    cols = FEATS[model]
    ytr = train.y.values.astype(int)
    med = train[cols].median()
    Xl = _lr_matrix(train, cols, med); sc = StandardScaler().fit(Xl)
    lr = LogisticRegression(**LR_PARAMS).fit(sc.transform(Xl), ytr)
    p_lr = lr.predict_proba(sc.transform(_lr_matrix(test, cols, med)))[:, 1]
    xg = XGBClassifier(**XGB_PARAMS).fit(train[cols].values.astype(np.float32), ytr)
    p_xg = xg.predict_proba(test[cols].values.astype(np.float32))[:, 1]
    return 0.5 * (p_lr + p_xg), p_lr, p_xg, dict(lr_coef=dict(zip(cols, np.round(lr.coef_[0], 4).tolist())),
                                                   xgb_gain={cols[int(k[1:])]: round(float(v), 3) for k, v in xg.get_booster().get_score(importance_type='gain').items()})

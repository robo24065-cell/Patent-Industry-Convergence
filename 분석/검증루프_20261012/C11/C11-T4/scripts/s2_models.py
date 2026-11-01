# 후보 A: 평가 원점 v의 PI·SB·M1(제공) 또는 M1_pub(공개 팔) 예측. 사용: python -I s2_models.py v world
#   world = p(제공 피처, PI·SB 포함) | fi | hs (P4b: 결과·위험은 제공, [K] 피처만 공개 배정) | pubw_fi | pubw_hs (P4c: 공개 세계)
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lib11 import *

CAND = 'A'
v = int(sys.argv[1]); world = sys.argv[2]
t_start = time.time()
KF = ['nk3', 'dens', 'maxphi', 'meanq', 'shq10', 'ck_log3', 'ck_gr', 'ck_nfirm', 'rfirm', 'rksic']
BASE = ['f_log3', 'f_log1', 'f_gr', 'is_cn']
if world == 'p':
    w = 'p'; feats = BASE + ['p_' + f for f in KF]; rk, yk = 'risk_p', 'y_p'
elif world in ('fi', 'hs'):
    w = world; feats = BASE + [w + '_' + f for f in KF] + [w + '_own_sh3', w + '_own_n1']; rk, yk = 'risk_p', 'y_p'
else:
    w = world.split('_')[1]; feats = BASE + [w + '_' + f for f in KF]; rk, yk = 'risk_' + w, 'y_' + w
need = ['cc', 'aid', 'k', rk, yk] + feats
if world == 'p' or world.startswith('pubw'):
    need = list(dict.fromkeys(need + [w + '_rfirm', w + '_rksic', w + '_nk3', w + '_ck_log3']))


def load(Y):
    d = pd.read_parquet(C + rf'\u_{CAND}_{Y}.parquet', columns=need)
    d = d[d[rk]].copy()
    d['fid'] = d.aid + np.where(d.cc == 'EP', 10_000_000, 0)
    d['Y'] = Y
    return d


tr = pd.concat([load(Y) for Y in ALL_Y if Y < v], ignore_index=True)
te = load(v)
ytr = tr[yk].values.astype(int)
rng = np.random.default_rng(SEED)
keep = (ytr == 1) | (rng.random(len(tr)) < 0.20)
wt = np.where(ytr == 1, 1.0, 5.0)
P = pd.DataFrame({'fid': te.fid.values, 'k': te.k.values, 'y': te[yk].values.astype(np.int8), 'cc': te.cc.values})
for sd in [20261013, 20261014, 20261015]:
    m = xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=20, n_jobs=2, random_state=sd, eval_metric='logloss')
    m.fit(tr.loc[keep, feats].values.astype(np.float32), ytr[keep], sample_weight=wt[keep])
    P[f'm_{sd}'] = m.predict_proba(te[feats].values.astype(np.float32))[:, 1]
    if sd == 20261013:
        gain = m.get_booster().get_score(importance_type='gain')
        gain = {feats[int(k[1:])]: float(g) for k, g in gain.items()}
P['m'] = P[[c for c in P.columns if c.startswith('m_')]].mean(1)


def logit(cols, transform=None):
    def fx(d):
        Z = d[cols].astype(float).copy()
        for c in cols:
            if c.endswith('_nk3'):
                Z[c] = np.log1p(Z[c])
        return Z
    Ztr = fx(tr); Zte = fx(te)
    mu = Ztr.mean()
    Ztr = Ztr.fillna(mu); Zte = Zte.fillna(mu)     # r_firm 결측(직전 원점 비활성) → 학습 평균
    sc = StandardScaler().fit(Ztr.values)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Ztr.values), ytr)
    return lr.decision_function(sc.transform(Zte.values))


if world == 'p' or world.startswith('pubw'):
    P['pi'] = logit([w + '_rfirm', w + '_rksic'])
    P['sb'] = logit(['f_log3', w + '_nk3', w + '_ck_log3'])
P.to_parquet(C + rf'\pred_{CAND}_{world}_{v}.parquet', index=False)
json.dump(dict(v=v, world=world, n_tr=int(len(tr)), n_tr_fit=int(keep.sum()), n_te=int(len(te)), pos_te=int(P.y.sum()), feats=feats, gain=gain,
               sec=round(time.time() - t_start, 1)), open(C + rf'\predlog_{CAND}_{world}_{v}.json', 'w'), indent=1)
out(v, world, len(te), int(P.y.sum()), round(time.time() - t_start, 1))

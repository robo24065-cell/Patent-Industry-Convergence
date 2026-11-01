"""반박 검사 2: XGB seed 민감도. 동결 spec 그대로, seed만 바꿔 S0·S1·S3·S4 최종(t0=2021) 재학습.
사용: python -I r0_seed.py 1,2   → r0_seed_{seeds}.json"""
import os, sys, json, time, numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, nbinom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
MY = os.path.dirname(os.path.abspath(__file__)); WD = os.path.dirname(MY)
R3 = os.path.abspath(os.path.join(WD, '..', '..', '..', '모델경쟁_20260927', 'T3_기술셀부상예측', 'R3'))
seeds = [int(x) for x in sys.argv[1].split(',')]
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8')); comps = SPEC['cfg']['components']
D = pd.read_parquet(os.path.join(MY, 'v0_D_final.parquet')); W = pd.read_parquet(os.path.join(MY, 'v0_W_final.parquet'))
SF = pd.read_parquet(os.path.join(WD, 'cache', 'scen_feats_fin.parquet'), columns=['cell', 't0', 'all_newapp_sh_S1', 'all_newapp_n_S1'])
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def run(D, W, seed, drop=()):
    tr = D[(D.t0 >= 2005) & (D.t0 <= 2018)]; te = D[D.t0 == 2021]; wtr = W[(W.t0 >= 2005) & (W.t0 <= 2018)]
    thr = D[D.yb == 1].groupby('t0').y.min(); ranks = []
    for c in comps:
        fs = [f for f in c['features'] if f not in drop]
        if c['type'] == 'logit':
            m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=c['C'], max_iter=3000)).fit(tr[fs].values, tr.yb.values)
            p = m.predict_proba(te[fs].values)[:, 1]
        else:
            w = wtr.copy(); prm = dict(c['params']); prm['seed'] = seed
            if c['type'] == 'xgbc':
                w['yb'] = (w.y >= w.t0.map(thr)).astype(int)
                X = np.vstack([tr[fs].values, w[fs].values]); yv = np.r_[tr.yb.values, w.yb.values]
                bst = xgb.train(prm, xgb.DMatrix(X, label=yv), c['rounds']); p = bst.predict(xgb.DMatrix(te[fs].values))
            else:
                X = np.vstack([tr[fs].values, w[fs].values]); yv = np.r_[tr.kr_fut.values, w.kr_fut.values]
                dm = xgb.DMatrix(X, label=yv); bst = xgb.train(prm, dm, c['rounds']); mu_tr = bst.predict(dm)
                ll = [(nbinom.logpmf(yv, 1 / a, (1 / a) / (1 / a + mu_tr)).sum(), a) for a in np.exp(np.linspace(np.log(0.01), np.log(5), 60))]
                a = max(ll)[1]; r = 1 / a
                tau = float(np.median([np.quantile(g.y.values, 0.9) for _, g in tr.groupby('t0')]))
                mu = bst.predict(xgb.DMatrix(te[fs].values)); need = np.ceil((te.kr_W.values + 1) * np.exp(tau) - 1 - 1e-9)
                p = nbinom.sf(need - 1, r, r / (r + mu))
        ranks.append(rankdata(p) / len(te))
    p = np.mean(ranks, 0)
    b = LogisticRegression(C=1e6, max_iter=1000).fit(tr[['kr_mom']].values, tr.yb.values).predict_proba(te[['kr_mom']].values)[:, 1]
    a, a0 = auc(te.yb.values, p), auc(te.yb.values, b)
    return dict(auc=float(a), auc_base=float(a0), skill=float((a - a0) / (1 - a0)))
AF = ['all_newapp_sh', 'all_newapp_n', 'kr_newapp_sh', 'kr_newk_sh', 'kr_napp', 'kr_newapp_n', 'kr_hhi', 'app_mom', 'app_size', 'big_sh', 'top1_sh']
def s1(X):
    n0 = len(X); X = X.merge(SF, on=['cell', 't0'], how='left'); assert len(X) == n0 and X.all_newapp_sh_S1.notna().all()
    X['all_newapp_sh'] = X.all_newapp_sh_S1.astype(np.float32); X['all_newapp_n'] = X.all_newapp_n_S1.astype(np.float32); return X
def s3(X):
    X = X.copy(); X['all_newapp_sh'] = X['kr_newapp_sh']; X['all_newapp_n'] = X['kr_newapp_n']; return X
D1, W1 = s1(D), s1(W); D3, W3 = s3(D), s3(W)
t = time.time(); out = {}
for sd in seeds:
    r = {'S0': run(D, W, sd), 'S1': run(D1, W1, sd), 'S3': run(D3, W3, sd), 'S4': run(D, W, sd, drop=AF)}
    out[sd] = r; print(sd, {k: round(v['skill'], 5) for k, v in r.items()}, round(time.time() - t), flush=True)
json.dump(out, open(os.path.join(MY, f"r0_seed_{'_'.join(map(str, seeds))}.json"), 'w'), indent=1)

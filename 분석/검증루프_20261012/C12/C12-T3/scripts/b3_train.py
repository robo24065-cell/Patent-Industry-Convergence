# -*- coding: utf-8 -*-
"""(B3) 민감도 재학습: 공식 동결 spec(final_spec_r3.json)·C09-T3 r0_seed.py와 같은 학습 규칙. 피처 값만 시나리오별로 바꾼다.
사용: python -B b3_train.py <seeds 쉼표> <시나리오 쉼표: S0,S1,S1a,S2,S3>   → cache/b3_<seeds>_<scen>.json, cache/b3_p_<scen>_s<seed>.parquet
공식 성적 불변: 민감도 전용 재학습."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata, nbinom, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
sys.stdout.reconfigure(encoding='utf-8')
seeds = [int(x) for x in sys.argv[1].split(',')]; scens = sys.argv[2].split(',')
SPEC = json.load(open(os.path.join(R3, 'final_spec_r3.json'), encoding='utf-8')); comps = SPEC['cfg']['components']
D0 = pd.read_parquet(os.path.join(C09V0, 'v0_D_final.parquet')); W0 = pd.read_parquet(os.path.join(C09V0, 'v0_W_final.parquet'))
F = pd.read_parquet(os.path.join(CACHE, 'b_feats.parquet'))
PIF = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']
P = pd.read_parquet(os.path.join(R3, 'preds', 'FINAL_R3_t0_2021.parquet'))
def with_scen(X, sc):
    if sc in ('S0', 'S3'): return X
    n0 = len(X); cols = ['ksic', 't0'] + [f'{k}__{sc}' for k in PIF]
    X = X.merge(F[cols], on=['ksic', 't0'], how='left'); assert len(X) == n0
    for k in PIF:
        X[k] = X[f'{k}__{sc}'].astype(X[k].dtype)
    return X.drop(columns=[f'{k}__{sc}' for k in PIF])
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
    return dict(auc=float(a), auc_base=float(a0), skill=float((a - a0) / (1 - a0))), te[['cell', 'yb']].assign(p=p, p_base=b)
t = time.time(); out = {}
for sc in scens:
    D, W = with_scen(D0, sc), with_scen(W0, sc)
    tr = D[(D.t0 >= 2005) & (D.t0 <= 2018)]; wtr = W[(W.t0 >= 2005) & (W.t0 <= 2018)]
    nanrate = {k: dict(train_D=float(tr[k].isna().mean()), train_W=float(wtr[k].isna().mean())) for k in PIF}
    for sd in seeds:
        r, pr = run(D, W, sd, drop=(PIF if sc == 'S3' else ()))
        mm = pr.merge(P[['cell', 'p', 'yb']].rename(columns={'p': 'p_off', 'yb': 'yb_off'}), on='cell')
        top = np.zeros(len(pr), bool); top[np.argsort(-pr.p.values)[:504]] = True
        r.update(hits_top10=int((top & (pr.yb.values == 1)).sum()), spearman_vs_official=float(spearmanr(mm.p, mm.p_off).correlation),
                 maxdiff_vs_official=float(np.abs(mm.p - mm.p_off).max()), yb_agree=bool((mm.yb == mm.yb_off).all()), nan_rate=nanrate)
        out[f'{sc}|{sd}'] = r
        pr.to_parquet(os.path.join(CACHE, f'b3_p_{sc}_s{sd}.parquet'), index=False)
        print(sc, sd, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'nan_rate'}, round(time.time() - t), flush=True)
json.dump(out, open(os.path.join(CACHE, f"b3_{'-'.join(map(str, seeds))}_{'-'.join(scens)}.json"), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

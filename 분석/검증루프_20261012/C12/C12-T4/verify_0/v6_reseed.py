# v6: 원점 v 하나에서 prov·free·hsc를 다른 시드(5,6)와 다른 학습 음성 재추출 시드로 다시 학습 → d_v의 시드 강건성
import sys, time, json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T4'
C = W + r'\cache'
v = int(sys.argv[1]); t = time.time()
PUBF = ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc', 'is_cn', 's_n5', 's_grow', 's_nfirm', 's_newrate',
        'p_sec', 'p_relmean', 'p_relmax', 'p_cls', 'p_sect', 'p_cf', 'pi_rfirm', 'pi_rsc']
CTX = ['dot', 'cos', 'adopt', 'entry', 'qmain', 'mom_c', 'sind_c', 'mom_kr', 'sind_kr', 'mom_5', 'sind_5']
arms = {s: PUBF + [f'{s}_{x}' for x in CTX] for s in ['prov', 'free', 'hsc']}
cols = ['y', 'w', 'Y'] + sorted(set(sum(arms.values(), [])))
def load(y, sub):
    f = pd.concat([pd.read_parquet(C + rf'\u_{cc}_{y}.parquet', columns=cols) for cc in ['CN', 'EP']], ignore_index=True)
    if sub:
        rng = np.random.default_rng(99 + y)
        keep = (f.y.values == 1) | (rng.random(len(f)) < 1 / 3)
        f = f[keep].copy(); f.loc[f.y == 0, 'w'] *= 3
    return f
tr = pd.concat([load(y, True) for y in [v - 3, v - 2, v - 1]], ignore_index=True); ev = load(v, False)
for c in cols[3:]:
    m = float(np.nanmean(tr[c].values)); tr[c] = tr[c].fillna(m); ev[c] = ev[c].fillna(m)
res = {}
for s, fc in arms.items():
    dtr = xgb.DMatrix(tr[fc].values, label=tr.y.values, weight=tr.w.values); dev = xgb.DMatrix(ev[fc].values)
    ps = []
    for sd in [5, 6]:
        prm = dict(objective='binary:logistic', tree_method='hist', eta=0.05, max_depth=5, min_child_weight=20,
                   subsample=0.8, colsample_bytree=0.7, reg_lambda=5, nthread=2, seed=sd)
        ps.append(xgb.train(prm, dtr, num_boost_round=300).predict(dev))
    res[s] = float(roc_auc_score(ev.y.values, np.mean(ps, 0), sample_weight=ev.w.values))
    print(s, res[s], round(time.time() - t, 1), flush=True)
ds = lambda a, b: (a - b) / (1 - b)
res['d_v'] = min(ds(res['prov'], res['free']), ds(res['prov'], res['hsc']))
print(res)
json.dump(res, open(W + rf'\verify_0\v6_out_{v}.json', 'w'), indent=1)

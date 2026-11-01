# C14-T4 b3: U6×Eb P0(b) — 실제 모형 예측 차이의 구조 확인(dyad) SE와 검정력 (PREREG §2.4)
# 학습 t0 2016–2022, 평가 t0 2023. 출력 cache/b3_p0b.json, cache/b3_pred_2023.parquet
import sys, os, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
from scipy.stats import norm
import xgboost as xgb
t = time.time()
TR = list(range(2016, 2023)); EV = 2023
U = pd.concat([pd.read_parquet(C + rf'\b2_u_{y}.parquet') for y in TR + [EV]], ignore_index=True)
BASE = [c for c in U.columns if c.startswith('b_')]
SX = {L: [c for c in U.columns if c.startswith(L + '_')] for L in ['prov', 'free', 'hsc']}
assert len(SX['prov']) == len(SX['free']) == len(SX['hsc'])
y = U.y.values.astype(int); tr = U.t0.isin(TR).values; ev = (U.t0 == EV).values
b1 = json.load(open(C + r'\b1_out.json', encoding='utf-8'))
E_POS = b1['Eb_pos']['lock_exp']; E_NEG = b1['Eb_neg']['lock_exp']


def fit_pred(cols, seed=SEED):
    m = xgb.XGBClassifier(tree_method='hist', max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=2, random_state=seed, eval_metric='logloss')
    m.fit(U.loc[tr, cols].values, y[tr])
    return m.predict_proba(U.loc[ev, cols].values)[:, 1]


pred = {'base': fit_pred(BASE)}
for L in SX:
    pred[L] = fit_pred(BASE + SX[L])
ye = y[ev]
E = U[ev].reset_index(drop=True)
w1 = np.ones(len(E))
preps = {k: AUCPrep(v, ye) for k, v in pred.items()}
auc = {k: p.auc(w1) for k, p in preps.items()}
pub = 'free' if auc['free'] >= auc['hsc'] else 'hsc'
dA = auc['prov'] - auc[pub]
out('AUC', auc, 'pub', pub, 'dA', dA)
pd.DataFrame({'key': E.key, 'key_g': E.key_g, 'y': ye, 'mf': E.mf_prov, 'mg': E.mg_prov, **{f'p_{k}': v for k, v in pred.items()}}).to_parquet(C + r'\b3_pred_2023.parquet', index=False)

# dyad 구조: 조직 노드(f, g), KSIC 노드(제공 최빈), KSIC 쌍 칸
nodes = pd.Index(pd.concat([E.key, E.key_g]).unique())
fi, gi = nodes.get_indexer(E.key), nodes.get_indexer(E.key_g)
kn = pd.Index(pd.concat([E.mf_prov, E.mg_prov]).astype(str).unique())
ki, kj = kn.get_indexer(E.mf_prov.astype(str)), kn.get_indexer(E.mg_prov.astype(str))
kc = np.where(E.mf_prov.astype(str) < E.mg_prov.astype(str), E.mf_prov.astype(str) + '|' + E.mg_prov.astype(str), E.mg_prov.astype(str) + '|' + E.mf_prov.astype(str))
kci = pd.factorize(kc)[0]
struct = {'n_eval': int(len(E)), 'pos': int(ye.sum()), 'neg': int((1 - ye).sum()), 'n_org_nodes': int(len(nodes)), 'n_ksic_nodes': int(len(kn)),
          'n_ksic_pair_cells': int(kci.max() + 1),
          'org_in_2plus_pairs_row_share': float(((pd.Series(np.r_[fi, gi]).value_counts() >= 2).reindex(fi).values | (pd.Series(np.r_[fi, gi]).value_counts() >= 2).reindex(gi).values).mean()),
          'org_modal_unique': True}
Bn = 400
rs = {k: np.random.default_rng(SEED + 10 + i) for i, k in enumerate(['Fp', 'Fc', 'Kp', 'Kc', 'iid'])}
D = {k: [] for k in ['Fp', 'Fc', 'Kp', 'Kc', 'iid']}
for b in range(Bn):
    wn = rs['Fp'].poisson(1.0, len(nodes)).astype(float); W = {'Fp': wn[fi] * wn[gi]}
    W['Fc'] = rs['Fc'].poisson(1.0, len(E)).astype(float)
    wk = rs['Kp'].poisson(1.0, len(kn)).astype(float); W['Kp'] = wk[ki] * wk[kj]
    W['Kc'] = rs['Kc'].poisson(1.0, kci.max() + 1).astype(float)[kci]
    W['iid'] = rs['iid'].poisson(1.0, len(E)).astype(float)
    for k, w in W.items():
        if w[ye == 1].sum() == 0 or w[ye == 0].sum() == 0:
            continue
        a = {m: preps[m].auc(w) for m in ('prov', 'free', 'hsc')}
        D[k].append(a['prov'] - max(a['free'], a['hsc']))
V = {k: float(np.var(v, ddof=1)) for k, v in D.items()}
V_F = V['Fp'] - 2 * V['Fc']; V_K = V['Kp'] - 2 * V['Kc']
SE = {'SE_' + k: float(np.sqrt(v)) for k, v in V.items()}
SE.update({'SE_Fdyad': float(np.sqrt(max(V_F, 0))), 'SE_Kdyad': float(np.sqrt(max(V_K, 0))), 'SE_primary': float(np.sqrt(max(V_F, V_K, 0)))})
npos, nneg = int(ye.sum()), int((1 - ye).sum())
scale = float(np.sqrt((1 / E_POS + 1 / E_NEG) / (1 / npos + 1 / nneg)))
pw = {k: float(norm.cdf(TARGET / (v * scale) - 1.96)) if v > 0 else None for k, v in SE.items()}
res = {'auc': auc, 'pub_arm': pub, 'dAUC_prov_pub': dA, 'dAUC_prov_base': auc['prov'] - auc['base'], 'struct': struct, 'B': Bn,
       'SE_dev': SE, 'scale_lock': scale, 'E_lock_pos': E_POS, 'E_lock_neg': E_NEG, 'power': pw,
       'ci_primary_normal': [dA - 1.96 * SE['SE_primary'], dA + 1.96 * SE['SE_primary']],
       'P0b_pass': bool(pw['SE_primary'] is not None and pw['SE_primary'] >= 0.80),
       'SE_lock_needed_for_0.80': TARGET / (1.96 + norm.ppf(0.8)), 'n_feat': {'base': len(BASE), 'sys': len(SX['prov'])}, 'secs': round(time.time() - t, 1)}
json.dump(res, open(C + r'\b3_p0b.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out(json.dumps(res, ensure_ascii=False, indent=1))

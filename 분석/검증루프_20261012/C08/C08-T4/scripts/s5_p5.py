# C08-T4 s5: P5 필수성(판정 밖 — P4에서 이미 기각). 단변량 S = |AUC(k_late1) − 0.5|, 평가 원점 2018–2022 평균.
# (a) 짝 순열 귀무 200회(시드 20261012+r), (b) 군집 부트스트랩 200회 CI. 공개 배정 TXT, CONC, HSC1(+보조 HSG3).
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import C
T = time.time()
EVAL = [2018, 2019, 2020, 2021, 2022]
cols = ['y', 'cd', 'origin'] + [f'k_late1_{a}' for a in ['prov', 'TXT', 'CONC', 'HSC1', 'HSG3']]
D = pd.concat([pd.read_parquet(os.path.join(C, f'feat_{Y}.parquet'), columns=cols) for Y in EVAL], ignore_index=True)


def auc_codes(code, y, K, w=None):
    w = np.ones(len(y)) if w is None else w
    wp = np.bincount(code, weights=w * y, minlength=K); wn = np.bincount(code, weights=w * (1 - y), minlength=K)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


rows, nullrows = [], []
G = {v: D[D.origin == v].reset_index(drop=True) for v in EVAL}
for A in ['TXT', 'CONC', 'HSC1', 'HSG3']:
    S_prov, S_pub, auc_p, auc_q = [], [], [], []
    prep = {}
    for v in EVAL:
        g = G[v]; y = g.y.values.astype(float)
        xp, xq = g['k_late1_prov'].values, g[f'k_late1_{A}'].values
        U, inv = np.unique(np.concatenate([xp, xq]), return_inverse=True)
        cp, cq = inv[:len(xp)], inv[len(xp):]
        prep[v] = (y, cp, cq, len(U))
        a1, a2 = auc_codes(cp, y, len(U)), auc_codes(cq, y, len(U))
        auc_p.append(a1); auc_q.append(a2)
    Sp = np.mean([abs(a - 0.5) for a in auc_p]); Sq = np.mean([abs(a - 0.5) for a in auc_q])
    d_obs = Sp - Sq
    dn = []
    for r in range(200):
        rng = np.random.default_rng(20261012 + r)
        s1, s2 = [], []
        for v in EVAL:
            y, cp, cq, K = prep[v]
            m = rng.random(len(y)) < 0.5
            s1.append(abs(auc_codes(np.where(m, cq, cp), y, K) - 0.5)); s2.append(abs(auc_codes(np.where(m, cp, cq), y, K) - 0.5))
        dn.append(np.mean(s1) - np.mean(s2))
    dn = np.array(dn)
    p = (1 + int((np.abs(dn) >= abs(d_obs)).sum())) / 201
    rows.append({'assign': A, 'judged': A != 'HSG3', 'auc_prov_mean': float(np.mean(auc_p)), 'auc_pub_mean': float(np.mean(auc_q)),
                 'S_prov': float(Sp), 'S_pub': float(Sq), 'delta_obs': float(d_obs), 'null_q025': float(np.quantile(dn, 0.025)),
                 'null_q975': float(np.quantile(dn, 0.975)), 'p_two_sided': p, 'pass_a': bool(p < 0.05 and d_obs > 0),
                 'auc_prov_by_origin': json.dumps([round(a, 5) for a in auc_p]), 'auc_pub_by_origin': json.dumps([round(a, 5) for a in auc_q])})
    nullrows.append(pd.DataFrame({'assign': A, 'r': np.arange(200), 'delta_null': dn}))
    print(rows[-1], round(time.time() - T, 1), flush=True)
# (b) 신호 존재: 군집 부트스트랩
cds = pd.Index(D.cd.unique())
rng = np.random.default_rng(20261012)
W = rng.multinomial(len(cds), np.ones(len(cds)) / len(cds), size=200).astype(float)
bs = []
for v in EVAL:
    g = G[v]; y = g.y.values.astype(float); gi = cds.get_indexer(g.cd)
    U, cp = np.unique(g['k_late1_prov'].values, return_inverse=True)
    bs.append([auc_codes(cp, y, len(U), W[b, gi]) for b in range(200)])
bm = np.array(bs).mean(axis=0)
auc_prov = rows[0]['auc_prov_mean']
B_ = {'auc_prov_mean': auc_prov, 'ci_lo': float(np.quantile(bm, 0.025)), 'ci_hi': float(np.quantile(bm, 0.975))}
B_['pass_b'] = bool(abs(auc_prov - 0.5) >= 0.02 and not (B_['ci_lo'] <= 0.5 <= B_['ci_hi']))
B_['direction_ok'] = bool(auc_prov > 0.5)
R = pd.DataFrame(rows)
LOG = {'P5b': B_, 'P5a_pass_all3': bool(R[R.judged].pass_a.all()), 'P5_pass(판정 밖)': bool(R[R.judged].pass_a.all() and B_['pass_b']), 'sec': round(time.time() - T, 1)}
R.to_csv(os.path.join(C, 's5_p5_result.csv'), index=False)
pd.concat(nullrows).to_csv(os.path.join(C, 's5_p5_null.csv'), index=False)
json.dump(LOG, open(os.path.join(C, 's5_p5_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(LOG, ensure_ascii=False, indent=1))

# C08-T4 verify_0 v4: v3에서 HGB Δskill +0.012가 나와 P4(XGB) 결론이 학습기에 민감한지 확인 — HGB 시드 2개(5, 6), 출원인 군집 부트스트랩 200회 CI.
import sys, os, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.ensemble import HistGradientBoostingClassifier
T = time.time()
CW = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C08\C08-T4\cache'
OUT = os.path.dirname(os.path.abspath(__file__))
M0 = ['age_y', 'util', 'days_to_due', 'overdue', 'n_due_win', 'next_ann', 'step', 'pat_late_n', 'pat_pay_n', 'last_lag', 'max_span',
      'own_late_r3', 'own_pay3', 'own_aband3', 'own_alive', 'app_type', 'n_app', 'n_transfer', 'n_lic', 'ipc_sec', 'nopay']
KF = ['k_late1_prov', 'k_trend_prov', 'k_aband1_prov']
D = pd.concat([pd.read_parquet(os.path.join(CW, f'feat_{Y}.parquet'), columns=['origin', 'y', 'cd'] + M0 + KF) for Y in range(2015, 2023)], ignore_index=True)
seed = int(sys.argv[1])
preds = []
for v in [2018, 2019, 2020, 2021, 2022]:
    tr = D[D.origin < v]; ev = D[D.origin == v]
    rng = np.random.default_rng(20261012)
    keep = (tr.y.values == 1) | (rng.random(len(tr)) < 0.3); trs = tr[keep]
    w = np.where(trs.y.values == 1, 1.0, 1 / 0.3)
    o = ev[['origin', 'y', 'cd']].copy()
    for nm, cols in {'M0': M0, 'M1': M0 + KF}.items():
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200, random_state=seed)
        m.fit(trs[cols], trs.y, sample_weight=w); o[nm] = m.predict_proba(ev[cols])[:, 1]
    preds.append(o); print(v, round(time.time() - T, 1), flush=True)
P = pd.concat(preds, ignore_index=True)
cl = pd.Index(P.cd.unique()); ci = cl.get_indexer(P.cd)
W = np.random.default_rng(20261012).multinomial(len(cl), np.ones(len(cl)) / len(cl), size=200).astype(float)
def aucs(g, c, gi):
    s = g[c].values; y = g.y.values.astype(float); o = np.argsort(s, kind='mergesort'); s, y, gg = s[o], y[o], gi[o]
    u, st = np.unique(s, return_index=True); grp = np.repeat(np.arange(len(st)), np.diff(np.append(st, len(s))))
    out = []
    for w in [np.ones(len(s))] + [W[b, gg] for b in range(200)]:
        p = np.bincount(grp, weights=w * y); n = np.bincount(grp, weights=w * (1 - y)); cn = np.cumsum(n) - n
        out.append((p * (cn + 0.5 * n)).sum() / (p.sum() * n.sum()))
    return np.array(out)
A0, A1 = [], []
for v, g in P.groupby('origin'):
    gi = ci[g.index.values]; A0.append(aucs(g, 'M0', gi)); A1.append(aucs(g, 'M1', gi))
A0, A1 = np.array(A0), np.array(A1)
sk = ((A1 - A0) / (1 - A0)).mean(axis=0)
res = {'seed': seed, 'dskill_mean': float(sk[0]), 'ci_lo': float(np.quantile(sk[1:], 0.025)), 'ci_hi': float(np.quantile(sk[1:], 0.975)),
       'auc_M0': float(A0[:, 0].mean()), 'auc_M1': float(A1[:, 0].mean())}
json.dump(res, open(os.path.join(OUT, f'v4_out_{seed}.json'), 'w'), indent=1); print(res)

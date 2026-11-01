# C08-T4 s4: P4 판정 — Δskill = (AUC1−AUC0)/(1−AUC0) 평가 원점 평균, 첫 출원인 코드 군집 부트스트랩 200회(시드 20261012).
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import C
from sklearn.metrics import average_precision_score
T = time.time()


def wauc(s, y, w):
    """가중 AUC(동점 0.5). s 점수, y 0/1, w 가중치."""
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    _, start = np.unique(s, return_index=True)
    grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    wp = np.bincount(grp, weights=w * y); wn = np.bincount(grp, weights=w * (1 - y))
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def boot_rows(P, cols, B=200, seed=20261012):
    cds = pd.Index(P.cd.unique()); ci = cds.get_indexer(P.cd)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(cds), np.ones(len(cds)) / len(cds), size=B).astype(float)   # 군집 가중치(원점 공통)
    res = {c: np.zeros((B, 0)) for c in cols}
    out = {c: [] for c in cols}
    for v, g in P.groupby('origin'):
        y = g.y.values.astype(float); gi = ci[g.index.values]
        for c in cols:
            s = g[c].values
            o = np.argsort(s, kind='mergesort'); ss = s[o]; yy = y[o]; gg = gi[o]
            _, start = np.unique(ss, return_index=True)
            grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(ss))))
            res_b = []
            for b in range(B):
                w = W[b, gg]
                wp = np.bincount(grp, weights=w * yy); wn = np.bincount(grp, weights=w * (1 - yy))
                cn = np.cumsum(wn) - wn
                res_b.append(float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum())))
            out[c].append(res_b)
    return {c: np.array(out[c]).T for c in cols}   # B × 원점


rows, LOG = [], {}
for arg in ['20261012', '20261013', '20261014', 'logit']:
    P = pd.read_parquet(os.path.join(C, f'p4_pred_{arg}.parquet')).reset_index(drop=True)
    a0 = P.groupby('origin').apply(lambda g: wauc(g.M0.values, g.y.values.astype(float), np.ones(len(g))))
    a1 = P.groupby('origin').apply(lambda g: wauc(g.M1.values, g.y.values.astype(float), np.ones(len(g))))
    pr0 = P.groupby('origin').apply(lambda g: average_precision_score(g.y, g.M0)); pr1 = P.groupby('origin').apply(lambda g: average_precision_score(g.y, g.M1))
    sk = ((a1 - a0) / (1 - a0))
    bt = boot_rows(P, ['M0', 'M1'])
    skb = ((bt['M1'] - bt['M0']) / (1 - bt['M0'])).mean(axis=1)
    r = {'model': 'XGB' if arg != 'logit' else 'logit', 'seed': arg, 'auc_M0_mean': float(a0.mean()), 'auc_M1_mean': float(a1.mean()),
         'dskill_mean': float(sk.mean()), 'dskill_ci_lo': float(np.quantile(skb, 0.025)), 'dskill_ci_hi': float(np.quantile(skb, 0.975)),
         'prauc_M0_mean': float(pr0.mean()), 'prauc_M1_mean': float(pr1.mean()),
         'dskill_by_origin': json.dumps({int(k): round(float(x), 5) for k, x in sk.items()}),
         'auc_M0_by_origin': json.dumps({int(k): round(float(x), 5) for k, x in a0.items()})}
    r['pass_seed'] = bool(r['dskill_ci_lo'] > 0 and r['dskill_mean'] >= 0.01)
    rows.append(r); print(r, round(time.time() - T, 1), flush=True)
R = pd.DataFrame(rows)
xg = R[R.model == 'XGB']
LOG['P4_pass'] = bool(xg.pass_seed.all())
LOG['note'] = '통과 = XGB 시드 3개 모두 CI 하한 > 0 이고 평균 Δskill ≥ 0.01. 로지스틱은 판정 밖.'
R.to_csv(os.path.join(C, 's4_p4_result.csv'), index=False)
json.dump(LOG, open(os.path.join(C, 's4_p4_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(LOG)

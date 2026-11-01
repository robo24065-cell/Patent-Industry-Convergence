import os, csv, datetime, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

LB = 'leaderboard.csv'
FIELDS = ['version', 'date', 'change', 'hypothesis', 'metric', 'value', 'baseline_value', 'skill', 'kept', 'note']


def p_at(y, p, frac=0.01):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p, kind='stable')[:k]
    return float(np.mean(np.asarray(y)[idx]))


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return dict(auc=roc_auc_score(y, p), prauc=average_precision_score(y, p), p1=p_at(y, p), base=y.mean())


def cluster_boot(y, p, cl, B=200, seed=0):
    """양도인(현재 권리자) 군집 부트스트랩 95% CI (AUC, PR-AUC)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    codes, uniq = pd.factorize(pd.Series(cl).fillna('NA'))
    order = np.argsort(codes, kind='stable'); cs = codes[order]
    starts = np.searchsorted(cs, np.arange(len(uniq))); ends = np.searchsorted(cs, np.arange(len(uniq)), side='right')
    aucs, prs = [], []
    nC = len(uniq)
    for b in range(B):
        pick = rng.integers(0, nC, nC)
        idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
        yy = y[idx]
        if yy.min() == yy.max(): continue
        aucs.append(roc_auc_score(yy, p[idx])); prs.append(average_precision_score(yy, p[idx]))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(prs, [2.5, 97.5])


def skill(auc, base_auc):
    return (auc - base_auc) / (1 - base_auc)


def log(version, change, hypothesis, value, baseline_value, kept, note, metric='AUC(val L2014+2015, y_MAIN)'):
    new = not os.path.exists(LB)
    rows = []
    if not new:
        rows = [r for r in csv.DictReader(open(LB, encoding='utf-8-sig')) if r['version'] != version]
    rows.append(dict(version=version, date=datetime.date.today().isoformat(), change=change, hypothesis=hypothesis, metric=metric,
                     value=round(value, 4), baseline_value=round(baseline_value, 4) if baseline_value is not None else '',
                     skill=round(skill(value, baseline_value), 4) if baseline_value is not None else '', kept=kept, note=note))
    with open(LB, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def seg_report(df, p, col, y='y_MAIN', min_n=500):
    out = []
    for k, d in df.assign(_p=p).groupby(col):
        if len(d) < min_n or d[y].nunique() < 2: continue
        out.append((k, len(d), int(d[y].sum()), round(d[y].mean(), 4), round(roc_auc_score(d[y], d._p), 4), round(d._p.mean(), 4)))
    return pd.DataFrame(out, columns=[col, 'n', 'pos', 'rate', 'auc', 'mean_p'])

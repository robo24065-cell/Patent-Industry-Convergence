# -*- coding: utf-8 -*-
"""T4 실험 러너. 사용: python run.py v0 [v1 ...]
검증: 기준일 L ∈ {2014-01-01, 2014-07-01, 2015-01-01, 2015-07-01}, 학습 레이블은 C=2014-01-01 시점 관측분만.
"""
import os, sys, time, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import t4lib as T

HERE = T.HERE
C_VAL = '2014-01-01'
EVAL_L = ['2014-01-01', '2014-07-01', '2015-01-01', '2015-07-01']


def semi(a, b):
    out, y, m = [], int(a[:4]), int(a[5:7])
    while f'{y:04d}-{m:02d}-01' <= b:
        out.append(f'{y:04d}-{m:02d}-01')
        m += 6
        if m > 12: m -= 12; y += 1
    return out


def train_landmarks(C, first='2005-01-01'):
    """L+3y+400일 <= C 인 반기 기준일"""
    Cd = np.datetime64(C)
    return [L for L in semi(first, C) if T.add_years(np.array([np.datetime64(L)]), 3)[0] + np.timedelta64(400, 'D') <= Cd]


def labels(L, pids, C=None):
    fn = os.path.join(T.CACHE, f'lab_{L.replace("-", "")}_{"full" if C is None else C.replace("-", "")}.npy')
    if os.path.exists(fn):
        y = np.load(fn)
        if len(y) == len(pids): return y
    y = T.label_full(L, pids) if C is None else T.label_asof(L, pids, C)
    np.save(fn, y.astype(float))
    return y.astype(float)


def get_frame(Ls, C=None):
    fs = []
    for L in Ls:
        f = T.features(L)
        y = labels(L, f.pid.values, C)
        f = f.assign(y=y, acd=T.load()['pt'].acd.values[f.pid.values])
        fs.append(f[np.isfinite(f.y)])
    return pd.concat(fs, ignore_index=True)


def metrics(y, p):
    return dict(auc=roc_auc_score(y, p), prauc=average_precision_score(y, p), brier=brier_score_loss(y, p), base_rate=float(np.mean(y)))


def boot_ci(df, pcol, cluster='acd', B=200, seed=0):
    rng = np.random.default_rng(seed)
    g = df.groupby(cluster).indices
    keys = np.array(list(g.keys()))
    vals = []
    idxs = [g[k] for k in keys]
    for _ in range(B):
        s = rng.integers(0, len(keys), len(keys))
        ii = np.concatenate([idxs[i] for i in s])
        vals.append(roc_auc_score(df.y.values[ii], df[pcol].values[ii]))
    return np.percentile(vals, [2.5, 97.5])


LB = os.path.join(HERE, 'leaderboard.csv')
LB_COLS = ['version', 'date', 'change', 'hypothesis', 'metric', 'value', 'baseline_value', 'skill', 'kept', 'note']


def log_lb(version, change, hyp, m, base_auc, kept, note):
    row = dict(version=version, date=time.strftime('%Y-%m-%d'), change=change, hypothesis=hyp, metric='AUC(val 2014-15)',
               value=round(m['auc'], 4), baseline_value=round(base_auc, 4) if base_auc else round(m['auc'], 4),
               skill=round((m['auc'] - base_auc) / (1 - base_auc), 4) if base_auc else 0.0, kept=kept,
               note=f"PR-AUC {m['prauc']:.4f} Brier {m['brier']:.4f} base {m['base_rate']:.4f}; " + note)
    df = pd.read_csv(LB) if os.path.exists(LB) else pd.DataFrame(columns=LB_COLS)
    df = df[df.version != version]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LB, index=False, encoding='utf-8-sig')
    print('LB>', row, flush=True)


def base_auc():
    if os.path.exists(LB):
        d = pd.read_csv(LB)
        r = d[d.version == 'v0']
        if len(r): return float(r.value.iloc[0])
    return None


def save_pred(version, ev, p):
    out = ev[['pid', 'L', 'ksic', 'y']].copy(); out['p'] = p
    out.to_parquet(os.path.join(HERE, 'preds', f'pred_{version}.parquet'), index=False)


def per_L(ev, p):
    return {L: round(roc_auc_score(ev.y[ev.L == L], p[(ev.L == L).values]), 4) for L in sorted(ev.L.unique())}


os.makedirs(os.path.join(HERE, 'preds'), exist_ok=True)

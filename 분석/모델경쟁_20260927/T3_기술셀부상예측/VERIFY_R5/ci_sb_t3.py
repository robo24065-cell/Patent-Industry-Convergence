# T3 강화 기준선 대비 skill의 IPC 메인그룹 군집 부트스트랩 95% CI(1,000회, 공식과 같은 군집). 산출 ci_sb_t3.json
import os, sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); out = {}
for tag in ('R3pub', 'R2app'):
    d = pd.read_parquet(os.path.join(HERE, f'sens_{tag}_scores.parquet')); sbs = [c for c in d.columns if c.startswith('SB')]
    blocks = [(g.mg.values, g.yb.values, g.p.values, {c: g[c].values for c in ['p_base'] + sbs}) for _, g in d.groupby('t0')]
    ums = np.unique(d.mg.values); gi = [pd.Series(np.arange(len(b[0]))).groupby(b[0]).apply(lambda s: s.values).to_dict() for b in blocks]
    rng = np.random.default_rng(11); R = {c: [] for c in ['p_base'] + sbs}
    for _ in range(1000):
        smp = rng.choice(ums, len(ums)); acc = {c: [] for c in R}
        for b, g in zip(blocks, gi):
            ii = np.concatenate([g[u] for u in smp if u in g]); y = b[1][ii]
            if y.min() == y.max(): continue
            am = roc_auc_score(y, b[2][ii])
            for c in R: a0 = roc_auc_score(y, b[3][c][ii]); acc[c].append((am - a0) / (1 - a0))
        for c in R: R[c].append(np.mean(acc[c]))
    out[tag] = {c: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for c, v in R.items()}
    print(tag, out[tag], flush=True)
json.dump(out, open(os.path.join(HERE, 'ci_sb_t3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

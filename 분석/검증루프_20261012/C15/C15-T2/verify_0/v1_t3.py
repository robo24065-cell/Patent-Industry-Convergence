# -*- coding: utf-8 -*-
import sys, json, numpy as np, pandas as pd
D = sys.argv[1]
def wauc(inv, U, y, w):
    wp = np.bincount(inv, w * y, U); wn = np.bincount(inv, w * (1 - y), U)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
out = {}
for b in range(10):
    P = pd.read_parquet(f'{D}/cache/t3/preds_b{b}_final.parquet')
    y = P.yb.values.astype(float); mg = P.mg.values
    ums, ginv = np.unique(mg, return_inverse=True)
    pre = {}
    for c in ['p', 'p_base', 'SB3']:
        u, inv = np.unique(P[c].values, return_inverse=True); pre[c] = (inv, len(u))
    one = np.ones(len(y))
    A = {c: wauc(*pre[c], y, one) for c in pre}
    sk = lambda a, a0: (a - a0) / (1 - a0)
    res = dict(auc=A['p'], m0=A['p_base'], sb3=A['SB3'], skill=sk(A['p'], A['p_base']), skill_sb3=sk(A['p'], A['SB3']),
               top100=int(y[np.argsort(-P.p.values, kind='stable')[:100]].sum()))
    for name, seed, base in [('ci_m0', 0, 'p_base'), ('ci_sb3', 11, 'SB3')]:
        rng = np.random.default_rng(seed); v = []
        for _ in range(1000):
            s = rng.choice(ums, len(ums))
            cnt = np.bincount(np.searchsorted(ums, s), minlength=len(ums)).astype(float)
            w = cnt[ginv]
            # 가중치 0 행 제외 효과는 가중합으로 자동 반영
            a = wauc(*pre['p'], y, w); a0 = wauc(*pre[base], y, w); v.append(sk(a, a0))
        res[name] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out[b] = res; print(b, {k: (round(x, 6) if isinstance(x, float) else x) for k, x in res.items()}, flush=True)
json.dump(out, open(f'{D}/verify_0/v1_t3_result.json', 'w'), indent=1)

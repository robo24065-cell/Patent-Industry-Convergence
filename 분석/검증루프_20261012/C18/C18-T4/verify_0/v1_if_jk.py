# C18-T4 verify_0 v1: 독립 SE 추정량(PREREG §7) — 작업 부트스트랩 코드(s3_boot) 미사용.
#  (1) 가중 AUC 영향함수(DeLong 배치값) 군집 합 해석식: V_k, V_f(K4: V_c), CGM, 층화(k 고정, 층 안 군집 편차)
#  (2) KSIC 하나 빼기 jackknife
# 같은 Bernoulli 부분표본(s3_boot와 같은 시드 규칙)에서 계산 → SE²(f)=a+b/f 적합 → 잠금 검정력.
# 사용: python -I v1_if_jk.py K v [L]   출력 verify_0/v1_{tag}.json
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
from lib18 import C, SEED, FRACS, E_LOCK, Z80, out
import numpy as np, pandas as pd
from scipy.stats import norm
t = time.time()
K = sys.argv[1]; v = int(sys.argv[2]); L = sys.argv[3] if len(sys.argv) > 3 else None
tag = f'{K}_{v}' + (f'_{L}' if L else '')
F0 = pd.read_parquet(C + (rf'\fr_{K}_{v}_{L}.parquet' if L else rf'\fr_{K}_{v}.parquet'))
best = json.load(open(C + r'\s2_point.json'))['best_public'][K + (f'_{L}' if L else '')]['best']


def auc_if(s, y, w):
    y = y.astype(bool)
    o = np.argsort(s, kind='mergesort'); ss = s[o]
    grp = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
    g = np.empty(len(s), np.int64); g[o] = grp
    ng = grp[-1] + 1
    wp = np.bincount(g[y], weights=w[y], minlength=ng); wn = np.bincount(g[~y], weights=w[~y], minlength=ng)
    Wp, Wn = wp.sum(), wn.sum()
    v10 = (np.cumsum(wn) - wn + 0.5 * wn) / Wn
    v01 = (Wp - np.cumsum(wp) + 0.5 * wp) / Wp
    th = float((wp * v10).sum() / Wp)
    psi = np.where(y, w * (v10[g] - th) / Wp, w * (v01[g] - th) / Wn)
    return th, psi, g, ng, y


def auc_from(g, ng, y, w):
    wp = np.bincount(g[y], weights=w[y], minlength=ng); wn = np.bincount(g[~y], weights=w[~y], minlength=ng)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def V(psi, cl):
    u, i = np.unique(cl, return_inverse=True); G = len(u)
    S = np.bincount(i, weights=psi)
    return float(G / (G - 1) * (S ** 2).sum())


def V_strat(psi, k, c):
    """층 k 고정, 층 안 군집 c 복원추출 분산: Σ_k m_k/(m_k−1) Σ_{c∈k} (S_c − S̄_k)²"""
    df = pd.DataFrame({'k': k, 'c': c, 'p': psi}).groupby(['k', 'c']).p.sum().reset_index()
    g = df.groupby('k').p
    m = g.transform('size').values; dev = df.p.values - g.transform('mean').values
    ok = m > 1
    return float((m[ok] / (m[ok] - 1) * dev[ok] ** 2).sum())


inner = {'K1': 'f', 'K2': 'f', 'K3': 'f', 'K4': 'c'}[K]
res = {'tag': tag, 'best': best, 'fracs': {}}
for fr in FRACS:
    if fr < 1.0:
        m = np.random.default_rng(SEED + 7 + int(fr * 10000)).random(len(F0)) < fr
        F = F0[m].reset_index(drop=True)
    else:
        F = F0
    y = F.y.values; w = F.w.values.astype(float)
    a1, p1, g1, n1, yb = auc_if(F.prov.values.astype(float), y, w)
    a2, p2, g2, n2, _ = auc_if(F[best].values.astype(float), y, w)
    psi = p1 - p2; d = a1 - a2
    k = F.k.values; cl_in = F[inner].values
    r = {'n': len(F), 'pos': int(y.sum()), 'd': d, 'IF_iid': float(np.sqrt((psi ** 2).sum())), 'IF_k': np.sqrt(V(psi, k)), 'IF_inner': np.sqrt(V(psi, cl_in))}
    if K == 'K1':
        Vk, Vs, Vks = V(psi, k), V(psi, F.s.values), V(psi, k * 1_000_000 + F.s.values)
        r['IF_CGM'] = float(np.sqrt(max(Vk + Vs - Vks, Vk, Vs)))
    elif K == 'K3':
        Vk, Vk2, Vkk = V(psi, k), V(psi, F.k2.values), V(psi, k * 1000 + F.k2.values)
        r['IF_CGM'] = float(np.sqrt(max(Vk + Vk2 - Vkk, Vk, Vk2)))
    elif K == 'K4':
        Vk, Vf, Vc = V(psi, k), V(psi, F.f.values), V(psi, F.c.values)
        r['IF_CGM'] = float(np.sqrt(max(Vf + Vk - Vc, Vf, Vk)))
        r['IF_f'] = float(np.sqrt(Vf))
    else:
        r['IF_CGM'] = r['IF_k']
    r['IF_strat'] = float(np.sqrt(V_strat(psi, k, cl_in)))
    # KSIC jackknife
    ks = np.unique(k); th = []
    for kk in ks:
        ww = w.copy(); ww[k == kk] = 0.0
        if ww[yb].sum() == 0 or ww[~yb].sum() == 0:
            continue
        th.append(auc_from(g1, n1, yb, ww) - auc_from(g2, n2, yb, ww))
    th = np.array(th); G = len(th)
    r['JK_k'] = float(np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum()))
    r['JK_G'] = G
    res['fracs'][str(fr)] = r
    out(tag, fr, json.dumps({a: (round(b, 6) if isinstance(b, float) else b) for a, b in r.items()}), round(time.time() - t, 1))


def fit(xs, ys):
    x = 1 / np.asarray(xs); yv = np.asarray(ys) ** 2
    a, b = np.linalg.lstsq(np.c_[np.ones_like(x), x], yv, rcond=None)[0]
    if a < 0:
        a = 0.0; b = float((x * yv).sum() / (x * x).sum())
    return float(a), float(b)


fL = E_LOCK[K] / res['fracs']['1.0']['pos']
est = {'D0_IF': lambda r: max(r['IF_CGM'], r['IF_k']), 'D0_JK': lambda r: r['JK_k'], 'D1a_IF': lambda r: max(r['IF_strat'], r['IF_inner']),
       'k_IF_only': lambda r: r['IF_k']}
res['lock'] = {}
for nm, fn in est.items():
    ses = [fn(res['fracs'][str(f)]) for f in FRACS]
    a, b = fit(FRACS, ses); sl = float(np.sqrt(a + b / fL))
    res['lock'][nm] = {'SE_full': ses[-1], 'SE_lock': sl, 'SE_inf': float(np.sqrt(a)),
                       'power_0.002': float(norm.cdf(0.002 / sl - 1.959964)), 'power_0.005': float(norm.cdf(0.005 / sl - 1.959964)), 'MDE': Z80 * sl}
    out(nm, json.dumps(res['lock'][nm]))
res['fL'] = fL; res['sec'] = round(time.time() - t, 1)
json.dump(res, open(os.path.join(HERE, f'v1_{tag}.json'), 'w'), indent=1, default=float)
out('done', res['sec'])

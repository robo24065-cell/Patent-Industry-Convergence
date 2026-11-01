# C18-T4 s3m: D3 (iii) 정보량 늘리기.
#  (a) 다중 원점 결합: 개발 원점 평균 Δpool의 SE(군집 가중을 원점 공통으로 공유). K1(2020–24), K4(2021–23, xgb·lr).
#  (b) 나라 풀링 투영(K1): 나라별 Δ의 영향함수 KSIC 기여 상관 ρ(CN, EP).
# 사용: python s3m_multi.py K1 | K4 xgb | K4 lr | rho
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
t = time.time()
Bn = 200
SCHF = [a.split('=')[1].split(',') for a in sys.argv[1:] if a.startswith('--sch=')]
SCHF = SCHF[0] if SCHF else None
arg = [a for a in sys.argv[1:] if not a.startswith('--')]


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
    return th, psi


if arg[0] == 'rho':
    best = json.load(open(C + r'\s2_point.json'))['best_public']['K1']['best']
    res = {}
    for v in [2020, 2021, 2022, 2023, 2024]:
        F = load_frame('K1', v)
        D = {}
        for c in [0, 1]:
            G = F[F.cc == c]
            y = G.y.values; w = G.w.values.astype(float)
            a1, p1 = auc_if(G.prov.values.astype(float), y, w); a2, p2 = auc_if(G[best].values.astype(float), y, w)
            psi = p1 - p2
            D[c] = pd.Series(psi).groupby(G.k.values).sum().reindex(range(50)).fillna(0).values
            res.setdefault(str(v), {})[f'd_{c}'] = a1 - a2
            res[str(v)][f'SE_k_{c}'] = float(np.sqrt((D[c] ** 2).sum() * 50 / 49))
            res[str(v)][f'SE_iid_{c}'] = float(np.sqrt((psi ** 2).sum()))
        cov = float((D[0] * D[1]).sum()); rho = cov / np.sqrt((D[0] ** 2).sum() * (D[1] ** 2).sum())
        res[str(v)]['rho_k'] = float(rho)
        out(v, json.dumps(res[str(v)]))
    res['rho_mean'] = float(np.mean([res[str(v)]['rho_k'] for v in [2020, 2021, 2022, 2023, 2024]]))
    json.dump(res, open(C + r'\s3m_rho.json', 'w'), indent=1)
    out('done rho', res['rho_mean'], round(time.time() - t, 1))
    sys.exit(0)

K = arg[0]; L = arg[1] if len(arg) > 1 else None
VS = [2020, 2021, 2022, 2023, 2024] if K == 'K1' else [2021, 2022, 2023]
best = json.load(open(C + r'\s2_point.json'))['best_public'][K + (f'_{L}' if L else '')]['best']
frs = [pd.read_parquet(C + (rf'\fr_{K}_{v}_{L}.parquet' if L else rf'\fr_{K}_{v}.parquet')) for v in VS]
if K == 'K1':
    for F in frs:
        F['ks'] = F.k * 1_000_000 + F.s
    SCH = {'k': ['k'], 's': ['s'], 'ks': ['ks'], 'f': ['f'], 'nest': ['k', 'f']}
else:
    SCH = {'k': ['k'], 'f': ['f'], 'c': ['c'], 'nest': ['k', 'c']}
preps = [(AUCPrep(F.prov.values, F.y.values), AUCPrep(F[best].values, F.y.values), F.w.values.astype(np.float64)) for F in frs]
pt = [a.auc(w) - b.auc(w) for a, b, w in preps]
res = {'K': K, 'L': L, 'best': best, 'origins': VS, 'd_v': pt, 'd_mean': float(np.mean(pt)), 'se': {}}
for j, (nm, cols) in enumerate(SCH.items()):
    if SCHF and nm not in SCHF:
        continue
    idx = []
    for col in cols:
        u = np.unique(np.concatenate([F[col].values for F in frs]))
        idx.append((len(u), [np.searchsorted(u, F[col].values) for F in frs]))
    rng = np.random.default_rng(SEED + 900 + j)
    dm = np.empty(Bn)
    for b in range(Bn):
        gw = [rng.poisson(1.0, G).astype(np.float64) for G, _ in idx]
        vals = []
        for i, (a, bb, w0) in enumerate(preps):
            g = np.ones(len(w0))
            for (G, ii), ww in zip(idx, gw):
                g = g * ww[ii[i]]
            w = w0 * g
            vals.append(a.auc(w) - bb.auc(w))
        dm[b] = np.mean(vals)
        assert time.time() - t < 530, 'time guard'
    res['se'][nm] = float(dm.std(ddof=1))
    out(K, L, nm, res['se'][nm], round(time.time() - t, 1))
res['sec'] = round(time.time() - t, 1)
json.dump(res, open(C + rf'\s3m_{K}' + (f'_{L}' if L else '') + ('_' + '-'.join(SCHF) if SCHF else '') + '.json', 'w'), indent=1)
out('done', json.dumps(res))

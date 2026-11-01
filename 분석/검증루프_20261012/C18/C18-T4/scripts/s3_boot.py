# C18-T4 s3: 후보 × 원점 × (학습기) 군집 부트스트랩 SE (PREREG §3–4). B=200, 부분표본 f ∈ FRACS.
# 사용: python s3_boot.py K v [L] [--full-only]
# 통계량: Δpool = AUC(prov) − AUC(best), Δstrat = 층(k) 안 AUC 차. 출력 cache/s3_{K}_{v}[_{L}].json (+ npz 반복값)
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
t = time.time()
K = sys.argv[1]; v = int(sys.argv[2])
L = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
full_only = '--full-only' in sys.argv
Bn = 200
tag = f'{K}_{v}' + (f'_{L}' if L else '')
F0 = pd.read_parquet(C + (rf'\fr_{K}_{v}_{L}.parquet' if L else rf'\fr_{K}_{v}.parquet'))
best = json.load(open(C + r'\s2_point.json'))['best_public'][K + (f'_{L}' if L else '')]['best']

# 군집 정의(후보별)
if K == 'K1':
    F0['ks'] = F0.k * 1_000_000 + F0.s
    SCHEMES = {'k': ('pois', 'k'), 's': ('pois', 's'), 'ks': ('pois', 'ks'), 'f': ('pois', 'f'),
               'nest': ('nest', 'k', 'f'), 'strat': ('strat', 'k', 'f')}
elif K == 'K2':
    SCHEMES = {'k': ('pois', 'k'), 'f': ('pois', 'f'), 'nest': ('nest', 'k', 'f'), 'strat': ('strat', 'k', 'f')}
elif K == 'K3':
    F0['kk'] = F0.k * 1000 + F0.k2
    SCHEMES = {'k': ('pois', 'k'), 'k2': ('pois', 'k2'), 'kk': ('pois', 'kk'), 'f': ('pois', 'f'),
               'nest': ('nest', 'k', 'f'), 'strat': ('strat', 'k', 'f')}
else:  # K4
    SCHEMES = {'k': ('pois', 'k'), 'f': ('pois', 'f'), 'c': ('pois', 'c'), 'nest': ('nest', 'k', 'c'), 'strat': ('strat', 'k', 'c')}


def inv(col):
    u, i = np.unique(col, return_inverse=True)
    return i.astype(np.int64), len(u)


def weights(F, sch, rng):
    """반복 B개의 단위 가중 생성기."""
    kind = sch[0]
    if kind == 'pois':
        i, G = inv(F[sch[1]].values)
        for _ in range(Bn):
            yield rng.poisson(1.0, G).astype(np.float64)[i]
    elif kind == 'nest':
        ik, Gk = inv(F[sch[1]].values); ic, Gc = inv(F[sch[2]].values)
        for _ in range(Bn):
            yield rng.poisson(1.0, Gk).astype(np.float64)[ik] * rng.poisson(1.0, Gc).astype(np.float64)[ic]
    else:  # strat: 층 k 고정, 층 안 군집을 복원추출(다항)
        ic, Gc = inv(F[sch[2]].values)
        ck = pd.Series(F[sch[1]].values).groupby(ic).first().values     # 군집의 층(군집 ⊂ 층 가정 검사)
        nlev = pd.Series(F[sch[1]].values).groupby(ic).nunique().values
        viol = int((nlev > 1).sum())
        lists = [np.where(ck == kk)[0] for kk in np.unique(ck)]
        for _ in range(Bn):
            cnt = np.zeros(Gc)
            for lst in lists:
                cnt += np.bincount(lst[rng.integers(0, len(lst), len(lst))], minlength=Gc)
            yield cnt[ic]
        weights.viol = viol


res = {'K': K, 'v': v, 'L': L, 'best': best, 'B': Bn, 'fracs': {}}
store = {}
fr_list = [1.0] if full_only else FRACS
for fi, fr in enumerate(fr_list):
    if fr < 1.0:
        m = np.random.default_rng(SEED + 7 + int(fr * 10000)).random(len(F0)) < fr
        F = F0[m].reset_index(drop=True)
    else:
        F = F0
    w0 = F.w.values.astype(np.float64)
    ap = AUCPrep(F.prov.values, F.y.values); ab = AUCPrep(F[best].values, F.y.values)
    sp = StratAUC(F.prov.values, F.y.values, F.k.values); sb = StratAUC(F[best].values, F.y.values, F.k.values)
    r = {'n': len(F), 'pos': int(F.y.sum()), 'dpool': ap.auc(w0) - ab.auc(w0), 'dstrat': sp.auc(w0) - sb.auc(w0), 'se': {}}
    for j, (nm, sch) in enumerate(SCHEMES.items()):
        rng = np.random.default_rng(SEED + 100 * (j + 1) + fi)
        dp = np.empty(Bn); ds = np.empty(Bn)
        for b, g in enumerate(weights(F, sch, rng)):
            w = w0 * g
            dp[b] = ap.auc(w) - ab.auc(w); ds[b] = sp.auc(w) - sb.auc(w)
        r['se'][nm] = {'pool': float(dp.std(ddof=1)), 'strat': float(ds.std(ddof=1))}
        if sch[0] == 'strat':
            r['strat_violations'] = int(getattr(weights, 'viol', -1))
        store[f'{fr}_{nm}_pool'] = dp; store[f'{fr}_{nm}_strat'] = ds
        assert time.time() - t < 530, 'time guard'
    res['fracs'][str(fr)] = r
    out(tag, fr, r['n'], r['pos'], round(r['dpool'], 5), json.dumps({k: round(x['pool'], 5) for k, x in r['se'].items()}), round(time.time() - t, 1))
res['sec'] = round(time.time() - t, 1)
sfx = '_full' if full_only else ''
json.dump(res, open(C + rf'\s3_{tag}{sfx}.json', 'w'), indent=1)
np.savez_compressed(C + rf'\s3_{tag}{sfx}.npz', **store)
out('done', tag, res['sec'])

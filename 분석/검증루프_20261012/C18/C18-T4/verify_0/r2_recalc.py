# C18-T4 verify_0 r2(재시도 반박): 작업 코드(lib18, s3_boot, v1_if_jk) 미사용 독립 재계산.
# (1) sklearn 가중 AUC로 Δ·최선 공개 팔·P0(b) 재계산  (2) 자체 영향함수로 군집 분산 V_k, V_f, V_c, 층화, NEST 해석식(V_k + 2V_inner)
# (3) 부분표본(f∈{1/16..1}, 각 f 6회 평균) IF 분산으로 SE²=a+b/f 적합 → 잠금 SE·검정력·MDE  (4) G_eff 재계산
# 읽기: C18-T4\cache\fr_*.parquet(읽기만). 쓰기: verify_0\r2_out.json
import sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
t0 = time.time()
C = sys.argv[1]; OUT = sys.argv[2]
ELOCK = {'K1': 15961.345, 'K2': 1981.51, 'K3': 3408.0, 'K4': 1960.4}
PUB = {'K1': ['free', 'hsc', 'scagg'], 'K2': ['free', 'hsc'], 'K3': ['free', 'hsc'], 'K4': ['free', 'hsc', 'off']}


def wauc(s, y, w):
    return roc_auc_score(y, s, sample_weight=w)


def if_auc(s, y, w):
    """가중 AUC 영향함수(단위별 기여, 합=0 근사). 동률 0.5."""
    y = y.astype(bool); P = w[y].sum(); N = w[~y].sum()
    sn = s[~y]; wn = w[~y]; o = np.argsort(sn); sn = sn[o]; cwn = np.r_[0, np.cumsum(wn[o])]
    sp = s[y]; wp = w[y]; o2 = np.argsort(sp); sp2 = sp[o2]; cwp = np.r_[0, np.cumsum(wp[o2])]
    lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right')
    phi = (cwn[lo] + 0.5 * (cwn[hi] - cwn[lo])) / N          # 양성: 음성보다 높을 확률
    lo2 = np.searchsorted(sp2, sn_orig := s[~y], 'left'); hi2 = np.searchsorted(sp2, sn_orig, 'right')
    psi = ((P - cwp[hi2]) + 0.5 * (cwp[hi2] - cwp[lo2])) / P   # 음성: 양성이 더 높을 확률
    A = (wp * phi).sum() / P
    out = np.empty(len(s))
    out[y] = wp * (phi - A) / P
    out[~y] = wn * (psi - A) / N
    return A, out


def var_cl(ifv, g):
    S = np.bincount(g, weights=ifv); G = (S != 0).sum() if len(S) > 1 else 1
    G = len(S)
    return S, float((S ** 2).sum() * G / (G - 1))


def var_strat(ifv, gin, gout_of_in):
    """층(gout) 고정, 층 안 군집(gin) 복원추출: Σ_k Σ_f (S_f − S̄_k)²."""
    S = np.bincount(gin, weights=ifv)
    k = gout_of_in
    m = np.bincount(k, weights=S) / np.bincount(k)
    return float(((S - m[k]) ** 2).sum())


def comps(F, best, K):
    y = F.y.values; w = F.w.values.astype(float)
    _, ip = if_auc(F.prov.values, y, w); _, ib = if_auc(F[best].values, y, w)
    d = ip - ib
    gk = pd.factorize(F.k)[0]
    inner = 'c' if K == 'K4' else 'f'
    gi = pd.factorize(F[inner])[0]
    r = {}
    _, r['k'] = var_cl(d, gk)
    _, r['inner'] = var_cl(d, gi)
    if K == 'K4':
        _, r['f'] = var_cl(d, pd.factorize(F.f)[0])
    if K == 'K1':
        _, r['s'] = var_cl(d, pd.factorize(F.s)[0]); _, r['ks'] = var_cl(d, pd.factorize(F.k.astype(np.int64) * 10_000_000 + F.s)[0])
    if K == 'K3':
        _, r['k2'] = var_cl(d, pd.factorize(F.k2)[0]); _, r['kk'] = var_cl(d, pd.factorize(F.k * 1000 + F.k2)[0])
    # 층화: inner ⊂ k 검사
    kof = pd.Series(gk).groupby(gi).agg(['first', 'nunique'])
    r['strat_viol'] = int((kof['nunique'] > 1).sum())
    r['strat'] = var_strat(d, gi, kof['first'].values)
    r['nest'] = r['k'] + 2 * r['inner']   # Poisson×Poisson 곱 가중의 해석적 기대 분산
    if K == 'K1':
        r['cgm'] = max(r['k'] + r['s'] - r['ks'], r['k'], r['s'])
    elif K == 'K2':
        r['cgm'] = r['k']
    elif K == 'K3':
        r['cgm'] = max(r['k'] + r['k2'] - r['kk'], r['k'], r['k2'])
    else:
        r['cgm'] = max(r['f'] + r['k'] - r['inner'], r['f'], r['k'])
    r['gate_prereg'] = max(r['cgm'], r['nest'], r['k'])   # SE_gate(D0·D2) 해석판
    r['gate_noNEST'] = max(r['cgm'], r['k'])
    r['d1a'] = max(r['strat'], r['f'] if K == 'K4' else r['inner'])
    return r


def fit(fs, vs):
    x = 1 / np.asarray(fs); yv = np.asarray(vs)
    a, b = np.linalg.lstsq(np.c_[np.ones_like(x), x], yv, rcond=None)[0]
    if a < 0:
        a = 0.0; b = float((x * yv).sum() / (x * x).sum())
    return float(a), float(b)


R = {'auc': {}, 'p0b': {}, 'power': {}, 'geff': {}}
# (1) AUC·Δ 전 원점
specs = [('K1', v, None) for v in range(2020, 2025)] + [('K1', v, 'lr') for v in range(2020, 2025)] + \
        [('K2', 2023, None), ('K3', 2023, None)] + [('K4', v, L) for v in (2021, 2022, 2023) for L in ('xgb', 'lr')]
frames = {}
for K, v, L in specs:
    nm = f'fr_{K}_{v}' + (f'_{L}' if L else '')
    F = pd.read_parquet(f'{C}\\{nm}.parquet')
    a = {m: wauc(F[m].values, F.y.values, F.w.values) for m in ['base', 'prov'] + PUB[K]}
    R['auc'][nm] = a
    if (K, v) in [('K1', 2024), ('K2', 2023), ('K3', 2023), ('K4', 2023)]:
        frames[(K, L)] = F
    # G_eff (양성 가중 몫 / 양성 건수 몫)
    pos = F[F.y == 1]
    sh = pos.groupby('k').w.sum() / pos.w.sum(); shc = pos.groupby('k').size() / len(pos)
    R['geff'][nm] = {'n_k': int(F.k.nunique()), 'G_eff_w': float(1 / (sh ** 2).sum()), 'G_eff_cnt': float(1 / (shc ** 2).sum()),
                     'top10_w': float(sh.nlargest(10).sum()), 'pos': int(F.y.sum()), 'n': len(F)}
    print(nm, {m: round(x, 5) for m, x in a.items()}, round(time.time() - t0, 1), flush=True)
# P0(b)
groups = {('K1', 'xgb'): [f'fr_K1_{v}' for v in range(2020, 2025)], ('K1', 'lr'): [f'fr_K1_{v}_lr' for v in range(2020, 2025)],
          ('K2', 'xgb'): ['fr_K2_2023'], ('K3', 'xgb'): ['fr_K3_2023'],
          ('K4', 'xgb'): [f'fr_K4_{v}_xgb' for v in (2021, 2022, 2023)], ('K4', 'lr'): [f'fr_K4_{v}_lr' for v in (2021, 2022, 2023)]}
best = {}
for (K, L), nms in groups.items():
    mean_auc = {m: np.mean([R['auc'][n][m] for n in nms]) for m in PUB[K]}
    b = max(mean_auc, key=mean_auc.get); best[(K, L)] = b
    d = {m: [R['auc'][n]['prov'] - R['auc'][n][m] for n in nms] for m in PUB[K]}
    R['p0b'][f'{K}_{L}'] = {'best': b, 'd_by_origin': d, 'd_mean': {m: float(np.mean(x)) for m, x in d.items()},
                            'frac_pos_vs_best': float(np.mean([x > 0 for x in d[b]])),
                            'pass_rule': bool(all(np.mean(x) > 0 for x in d.values()) and np.mean([x > 0 for x in d[b]]) >= {'K1': .8, 'K4': 2 / 3}.get(K, 1.0) - 1e-9)}
    print('P0b', K, L, b, {m: round(float(np.mean(x)), 5) for m, x in d.items()}, R['p0b'][f'{K}_{L}']['pass_rule'], flush=True)

# (2)(3) 주 원점 IF 분산·부분표본 적합·검정력
FR = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
for (K, L), F in frames.items():
    LL = L or 'xgb'; b = best[(K, LL)]
    full = comps(F, b, K)
    per = {f: [] for f in FR}
    for f in FR:
        reps = 1 if f == 1.0 else 6
        for r_ in range(reps):
            if f < 1.0:
                m = np.random.default_rng(777 + 31 * r_ + int(f * 1000)).random(len(F)) < f
                Fs = F[m].reset_index(drop=True)
            else:
                Fs = F
            per[f].append(comps(Fs, b, K))
        if time.time() - t0 > 500:
            raise SystemExit('time guard')
    fL = ELOCK[K] / int(F.y.sum())
    pw = {}
    for est in ['k', 'cgm', 'nest', 'gate_prereg', 'gate_noNEST', 'strat', 'd1a']:
        vs = [np.mean([c[est] for c in per[f]]) for f in FR]
        a, bb = fit(FR, vs)
        se_l = np.sqrt(a + bb / fL)
        pw[est] = {'SE_full': float(np.sqrt(full[est])), 'a': a, 'b': bb, 'SE_lock': float(se_l), 'SE_inf': float(np.sqrt(a)),
                   'MDE_inf': float(2.8016 * np.sqrt(a)), 'MDE_lock': float(2.8016 * se_l),
                   'power_0.002': float(norm.cdf(0.002 / se_l - 1.959964)), 'power_0.005': float(norm.cdf(0.005 / se_l - 1.959964))}
    R['power'][f'{K}_{LL}'] = {'best': b, 'fL': fL, 'strat_viol': full['strat_viol'], 'est': pw}
    print('POW', K, LL, b, 'fL %.4f' % fL, {e: (round(x['SE_full'], 5), round(x['SE_lock'], 5), round(x['power_0.002'], 3), round(x['power_0.005'], 3), round(x['MDE_inf'], 4)) for e, x in pw.items()}, round(time.time() - t0, 1), flush=True)
R['sec'] = round(time.time() - t0, 1)
json.dump(R, open(OUT, 'w'), indent=1, default=float)
print('done', R['sec'])

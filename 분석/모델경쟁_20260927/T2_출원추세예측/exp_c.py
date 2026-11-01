from lab2 import *
import final_run as FR
M = {k: member_cache(k, v) for k, v in BASE.items()}
names = list(M)

# (1) EB 수정: 기울기 분산 = var(diff)/k^2 (수준잡음 모형; 끝점 기울기) + 국가합 비감쇠·개별편차 감쇠 경로(topdown형)
def eb2(E, o, H, k=8, phi=0.8, pt=1.0, vdiv='k2'):
    n = E['n']; CCI = E['CCI']; L = np.log1p(n)
    g = (L[:, -1] - L[:, -1 - k]) / k
    Ct = np.log(np.stack([n[CCI == j].sum(0) for j in range(5)]))
    gc = ((Ct[:, -1] - Ct[:, -1 - k]) / k)[CCI]
    d = np.diff(L[:, -k - 1:], axis=1)
    s2 = d.var(1, ddof=1) / (k * k if vdiv == 'k2' else k)
    B = np.zeros(len(n))
    for j in range(5):
        m = CCI == j
        tau2 = max(np.var(g[m] - gc[m], ddof=1) - s2[m].mean(), 1e-6)
        B[m] = tau2 / (tau2 + s2[m])
    st1 = np.cumsum(pt ** np.arange(1, H + 1)); st2 = np.cumsum(phi ** np.arange(1, H + 1))
    return np.expm1(L[:, -1:] + gc[:, None] * st1[None, :] + (B * (g - gc))[:, None] * st2[None, :])
FR.MEMBER_FNS['eb2'] = eb2
for kw in [{'vdiv': 'k2', 'pt': 0.8}, {'vdiv': 'k2', 'pt': 1.0}, {'vdiv': 'k', 'pt': 1.0}]:
    E2 = member_cache('eb2', kw); rep(rows_from(E2), f'eb2 {kw}')
    rep(rows_from(combine([M['ridge_lead'], M['flow_um'], M['topdown'], E2])), f'  ens+eb2 {kw}')

# (2) 스태킹: 검증원점 o, h별로 내부 원점 t∈[2008, o-h] (목표 t+h ≤ o) 에서 단체(simplex) 가중 격자 탐색
grid = [w for w in itertools.product(np.arange(0, 1.0001, 0.05), repeat=3) if abs(sum(w) - 1) < 1e-9]
def stack(per_cc=False):
    out = {}
    for o in FR.VAL_ORIGINS:
        F = np.zeros((250, H))
        for h in range(1, H + 1):
            inner = [t for t in range(2008, o - h + 1)]
            A = []; Yt = []; SC = []
            for t in inner:
                A.append(np.stack([M[m][t][:, h - 1] for m in names], 1)); Yt.append(D['n'][:, t - FR.Y0 + h])
                SC.append(FR.mase_scale(D['n'][:, :t - FR.Y0 + 1]))
            A = np.concatenate(A); Yt = np.concatenate(Yt); SC = np.concatenate(SC)
            cci = np.tile(D['CCI'], len(inner))
            groups = [np.arange(5)] if not per_cc else [[j] for j in range(5)]
            for gset in groups:
                msk = np.isin(cci, gset)
                errs = [np.mean(np.abs(A[msk] @ np.array(w) - Yt[msk]) / SC[msk]) for w in grid]
                wb = np.array(grid[int(np.argmin(errs))])
                rows = np.isin(D['CCI'], gset)
                F[rows, h - 1] = np.stack([M[m][o][rows, h - 1] for m in names], 1) @ wb
                if o == 2015: print('  o2015 h', h, 'grp', gset if per_cc else 'all', 'w', wb.round(2), 'n_inner', len(inner))
        out[o] = F
    return out
rep(rows_from(stack(False)), 'stack pooled')
rep(rows_from(stack(True)), 'stack per-cc')

from lab2 import *
import final_run as FR
M = {k: member_cache(k, v) for k, v in BASE.items()}
# EB 진단: 수축계수 분포
o = 2015; E = FR.truncate(D, o); n = E['n']; L = np.log1p(n); k = 8
g = (L[:, -1] - L[:, -1 - k]) / k
d = np.diff(L[:, -k - 1:], axis=1); s2 = d.var(1, ddof=1) / k
for j, c in enumerate(FR.CCS):
    m = E['CCI'] == j
    Ct = np.log(n[m].sum(0)); gc = (Ct[-1] - Ct[-1 - k]) / k
    tau2 = max(np.var(g[m] - gc, ddof=1) - s2[m].mean(), 1e-6)
    B = tau2 / (tau2 + s2[m])
    print(c, 'tau2', round(tau2, 5), 'mean s2', round(s2[m].mean(), 5), 'B median', round(np.median(B), 3), 'B p10', round(np.quantile(B, .1), 3))
# 다중 창 기울기 (k=5,8 평균)
F2 = member_cache('flow_um', dict(BASE['flow_um'], k=(5, 8)))
T2 = member_cache('topdown', dict(BASE['topdown'], kt=(5, 8), ks=(5, 8)))
rep(rows_from(F2), 'flow k(5,8)'); rep(rows_from(T2), 'topdown k(5,8)')
rep(rows_from(combine([M['ridge_lead'], F2, T2])), 'ens multi-k')
# damped ETS(로그, 2005~) 구성원
ET = member_cache('etslog', {'start': 2005}); rep(rows_from(ET), 'etslog2005')
rep(rows_from(combine(list(M.values()) + [ET])), 'ens+etslog')
# 재조정: 앙상블 국가합을 topdown 국가합에 맞춤
ens = combine(list(M.values()))
rec = {}
for o in ORIG_ALL:
    F = ens[o].copy()
    for j in range(5):
        m = D['CCI'] == j
        F[m] *= (M['topdown'][o][m].sum(0) / F[m].sum(0))[None, :]
    rec[o] = F
rep(rows_from(rec), 'ens recon->topdown totals')

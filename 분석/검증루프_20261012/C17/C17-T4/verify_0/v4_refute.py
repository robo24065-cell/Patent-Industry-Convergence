# C17-T4 verify v4 (재시도 반박): s5/s6/v1/v2 코드 미사용. sklearn AUC, 군집 jackknife(삭제-1), 새 시드 부트스트랩, P0(a) pandas 재계산.
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
W = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C17/C17-T4'
res = {}
# 1) 개발 3원점 AUC, 최선 공개 팔
for L in ('xgb', 'lr'):
    tab = {}
    for Y in (2021, 2022, 2023):
        P = pd.read_parquet(W + f'/cache/pred_{Y}_{L}.parquet')
        tab[Y] = {a: roc_auc_score(P.y, P[a]) for a in ('base', 'prov', 'free', 'hsc', 'off')}
        tab[Y]['n'] = len(P); tab[Y]['pos'] = int(P.y.sum())
    mean = {a: float(np.mean([tab[Y][a] for Y in tab])) for a in ('base', 'prov', 'free', 'hsc', 'off')}
    best = max(('free', 'hsc', 'off'), key=lambda a: mean[a])
    res[L] = {'by_origin': tab, 'mean': mean, 'best_public': best,
              'dAUC_by_origin_prov_minus_best': {Y: tab[Y]['prov'] - tab[Y][best] for Y in tab}}
# 2) P0(a) pandas 재계산
U = pd.read_parquet(W + '/cache/units.parquet', columns=['s', 'D', 'y'])
dv = U[U.D < 20250101]
lk = U[(U.D >= 20250101) & (U.D <= 20250117)]
base = {s: float(dv[(dv.s == s) & (dv.D >= 20230101) & (dv.D <= 20241115)].y.mean()) for s in (2, 3)}
nl = {s: int((lk.s == s).sum()) for s in (2, 3)}
res['P0a'] = {'n_lock': nl, 'base': base, 'exp_pos': sum(nl[s] * base[s] for s in (2, 3)),
              'lock_y_nonnull': int(lk.y.notna().sum())}
# 3) P0(b) SE: 삭제-1 군집 jackknife + 새 시드 부트스트랩(KSIC)
P = pd.read_parquet(W + '/cache/pred_2023_xgb.parquet')
best = res['xgb']['best_public']


def d_auc(d):
    return roc_auc_score(d.y, d.prov) - roc_auc_score(d.y, d[best])


def jk(d, col):
    g = d[col].values; ids = np.unique(g); G = len(ids)
    th = np.array([d_auc(d[g != k]) for k in ids])
    return float(np.sqrt((G - 1) / G * np.sum((th - th.mean()) ** 2)))


def bs_ksic(d, B, rng):
    ids, inv = np.unique(d.k_prov.values, return_inverse=True); G = len(ids)
    out = []
    for _ in range(B):
        pick = rng.integers(0, G, G)
        dd = pd.concat([d[inv == k] for k in pick], ignore_index=True)
        out.append(d_auc(dd))
    return float(np.std(out, ddof=1))


res['full2023'] = {'dAUC': d_auc(P), 'se_jk_ksic': jk(P, 'k_prov')}
rng = np.random.default_rng(777)
rows = []
for i in range(20):
    d = pd.concat([P[P.s == s].sample(n, random_state=5000 + i) for s, n in ((2, 4171), (3, 2756))], ignore_index=True)
    r = {'dAUC': d_auc(d), 'se_jk_ksic': jk(d, 'k_prov')}
    if i < 5:
        r['se_bs_ksic'] = bs_ksic(d, 100, rng)
    rows.append(r)
R = pd.DataFrame(rows)
res['sub'] = R.mean().to_dict()
se = float(R.se_jk_ksic.mean())
res['power_0.002_jk_ksic_only'] = float(norm.cdf(0.002 / se - 1.96))
res['power_0.002_if_se_halved'] = float(norm.cdf(0.002 / (se / 2) - 1.96))
res['se_needed_for_power_0.8'] = 0.002 / (1.96 + norm.ppf(0.8))
json.dump(res, open(W + '/verify_0/v4_out.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))

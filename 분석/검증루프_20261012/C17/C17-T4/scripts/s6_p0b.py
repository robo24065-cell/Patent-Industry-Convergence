# C17-T4 s6: P0(b) 검정력. 원점 2023 XGB 예측(PROV 대 최선 공개 팔), 잠금 크기 부분표본 20개 × 군집 부트스트랩 B=200.
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib17 import *
import numpy as np, pandas as pd
from scipy.stats import norm
t_start = time.time()
LEARN = sys.argv[1] if len(sys.argv) > 1 else 'xgb'
# 최선 공개 팔: 개발 3원점 평균 AUC
mean_auc = {}
for arm in ['free', 'hsc', 'off', 'prov', 'base']:
    mean_auc[arm] = float(np.mean([json.load(open(C + rf'\s5_{Y}_{LEARN}_base-off.json'))['auc'][arm] for Y in (2021, 2022, 2023)]))
best = max(['free', 'hsc', 'off'], key=lambda a: mean_auc[a])
P = pd.read_parquet(C + rf'\pred_2023_{LEARN}.parquet')
s3 = json.load(open(C + r'\s3b_checks.json'))
NL = {int(k): v for k, v in s3['P0a']['n_lock_by_s'].items()}


def wauc(p, y, w):
    u, inv = np.unique(p, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(u))
    wn = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    cn = np.cumsum(wn) - wn
    num = np.sum(wp * (cn + 0.5 * wn))
    return num / (wp.sum() * wn.sum())


def dauc(d, w):
    return wauc(d['prov'].values, d.y.values, w) - wauc(d[best].values, d.y.values, w)


def boot(d, col, B, rng):
    ids, inv = np.unique(d[col].values, return_inverse=True)
    G = len(ids); out_ = np.empty(B)
    for b in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        out_[b] = dauc(d, cnt[inv])
    return out_


P['cell'] = P.firm + '|' + P.k_prov
rng = np.random.default_rng(SEED)
# 구조 진단(2023 전체)
fk = P.groupby('firm').k_prov.agg(lambda s: s.value_counts(normalize=True).iloc[0])
fn = P.groupby('firm').size()
multi = fn[fn >= 2].index
diag = {'n_units': len(P), 'n_firms': int(P.firm.nunique()), 'n_ksic': int(P.k_prov.nunique()), 'n_cells': int(P.cell.nunique()),
        'firms_ge2_units': int(len(multi)), 'median_modal_ksic_share_firms_ge2': float(fk[multi].median()),
        'unit_weighted_modal_share': float((fk[P.firm].values).mean()),
        'firm_ksic_count_median_ge2': float(P[P.firm.isin(multi)].groupby('firm').k_prov.nunique().median()),
        'top10_firm_unit_share': float(fn.sort_values(ascending=False).head(10).sum() / len(P)),
        'top10_ksic_pos_share': float(P[P.y == 1].k_prov.value_counts().head(10).sum() / P.y.sum())}
full = {'dAUC_full': float(dauc(P, np.ones(len(P))))}
for col in ['firm', 'k_prov', 'cell']:
    full[f'se_full_{col}'] = float(boot(P, col, 200, rng).std(ddof=1))
rows = []
for i in range(20):
    parts = [P[P.s == s].sample(NL[s], random_state=SEED + i) for s in (2, 3)]
    d = pd.concat(parts, ignore_index=True)
    r = {'i': i, 'n': len(d), 'pos': int(d.y.sum()), 'dAUC': float(dauc(d, np.ones(len(d))))}
    V = {}
    for col in ['firm', 'k_prov', 'cell']:
        bs = boot(d, col, 200, rng); V[col] = bs.var(ddof=1); r[f'se_{col}'] = float(np.sqrt(V[col]))
    vc = V['firm'] + V['k_prov'] - V['cell']
    r['se_cgm'] = float(np.sqrt(max(vc, max(V['firm'], V['k_prov']))))
    r['cgm_raw_neg'] = bool(vc < max(V['firm'], V['k_prov']))
    r['se_gate'] = max(r['se_cgm'], r['se_k_prov'])
    rows.append(r)
R = pd.DataFrame(rows)
se_gate = float(R.se_gate.mean())
power = float(norm.cdf(0.002 / se_gate - 1.96))
res = {'learner': LEARN, 'mean_auc_dev': mean_auc, 'best_public': best, 'diag': diag, 'full2023': full,
       'sub_mean': R.mean(numeric_only=True).to_dict(), 'sub_sd_dAUC': float(R.dAUC.std(ddof=1)),
       'SE_gate': se_gate, 'SE_cgm_mean': float(R.se_cgm.mean()), 'SE_ksic_mean': float(R.se_k_prov.mean()), 'SE_firm_mean': float(R.se_firm.mean()),
       'power_0.002': power, 'P0b_pass': power >= 0.80,
       'power_at_observed_full_dAUC': float(norm.cdf(abs(full['dAUC_full']) / se_gate - 1.96)), 'sec': round(time.time() - t_start, 1)}
R.to_csv(C + rf'\s6_p0b_sub_{LEARN}.csv', index=False)
json.dump(res, open(C + rf'\s6_p0b_{LEARN}.json', 'w'), indent=1)
out(json.dumps(res, indent=1))

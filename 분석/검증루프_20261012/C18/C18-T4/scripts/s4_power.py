# C18-T4 s4: 검정력 지도 조립(PREREG §3–4). s3_* / s3m_* 캐시만 읽는다. 출력 cache/s4_power.json, power_map.csv
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
from scipy.stats import norm

S2 = json.load(open(C + r'\s2_point.json'))
FRS = ['0.0625', '0.125', '0.25', '0.5', '1.0']


def cgm(K, se):
    V = {k: x ** 2 for k, x in se.items()}
    if K == 'K1':
        return np.sqrt(max(V['k'] + V['s'] - V['ks'], V['k'], V['s']))
    if K == 'K2':
        return np.sqrt(V['k'])
    if K == 'K3':
        return np.sqrt(max(V['k'] + V['k2'] - V['kk'], V['k'], V['k2']))
    return np.sqrt(max(V['f'] + V['k'] - V['c'], V['f'], V['k']))


def se_design(K, r, design):
    pool = {k: x['pool'] for k, x in r['se'].items()}
    strat = {k: x['strat'] for k, x in r['se'].items()}
    if design in ('D0', 'D2'):
        return max(cgm(K, pool), pool['nest'], pool['k']), {'CGM': cgm(K, pool), 'NEST': pool['nest'], 'k': pool['k']}
    if design == 'D1a':
        return max(pool['strat'], pool['f']), {'strat': pool['strat'], 'f': pool['f']}
    if design == 'D1b':
        return max(strat['strat'], strat['f']), {'strat': strat['strat'], 'f': strat['f']}


def fit(xs, ys):
    """SE^2 = a + b/f, 가중 없는 최소제곱, a >= 0."""
    x = 1 / np.asarray(xs); y = np.asarray(ys) ** 2
    A = np.c_[np.ones_like(x), x]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if a < 0:
        a = 0.0; b = float((x * y).sum() / (x * x).sum())
    return float(a), float(b)


def pw(d, se):
    return float(norm.cdf(d / se - 1.959964))


TARGET = {'D0': 0.002, 'D1a': 0.002, 'D1b': 0.002, 'D2': 0.005}
rows = []
res = {}
MAINS = [('K1', 2024, None), ('K1', 2024, 'lr'), ('K2', 2023, None), ('K3', 2023, None), ('K4', 2023, 'xgb'), ('K4', 2023, 'lr')]
for K, v, L in MAINS:
    tag = f'{K}_{v}' + (f'_{L}' if L else '')
    J = json.load(open(C + rf'\s3_{tag}.json'))
    full = J['fracs']['1.0']
    pos = full['pos']; fL = E_LOCK[K] / pos
    for D, tgt in TARGET.items():
        ses = [se_design(K, J['fracs'][f], D)[0] for f in FRS]
        a, b = fit([float(f) for f in FRS], ses)
        se_lock = np.sqrt(a + b / fL); se_inf = np.sqrt(a)
        comp_full = se_design(K, full, D)[1]
        obs = full['dstrat'] if D == 'D1b' else full['dpool']
        r = {'cand': K, 'learner': L or 'xgb', 'design': D, 'origin': v, 'target': tgt, 'obs_delta_dev': obs,
             'SE_full': ses[-1], **{f'SE_full_{k}': x for k, x in comp_full.items()}, 'fit_a': a, 'fit_b': b,
             'fL': fL, 'fL_note': ('외삽(>1)' if fL > 1 else ('하한 밖 소폭 외삽(<1/16)' if fL < 1 / 16 else '보간')),
             'SE_lock': se_lock, 'SE_inf': se_inf, 'power_lock': pw(tgt, se_lock), 'MDE_lock': Z80 * se_lock, 'MDE_inf': Z80 * se_inf,
             'need_lock_multiple_for_0.8': (float(b / ((tgt / Z80) ** 2 - a) / fL) if (tgt / Z80) ** 2 > a else float('inf')),
             'E_lock_pos': E_LOCK[K], 'P0a': E_LOCK[K] >= 300, 'G1': pw(tgt, se_lock) >= 0.8,
             'ses_by_frac': dict(zip(FRS, ses))}
        rows.append(r)
# 안정성: 다른 개발 원점(전체 규모만) — 주 원점의 SE_lock/SE_full 비를 곱해 환산(보고용)
stab = []
for fn in sorted(glob.glob(C + r'\s3_*_full.json')):
    J = json.load(open(fn)); K = J['K']; L = J['L']
    for D, tgt in TARGET.items():
        se_f = se_design(K, J['fracs']['1.0'], D)[0]
        main = [r for r in rows if r['cand'] == K and r['design'] == D and r['learner'] == (L or 'xgb')][0]
        ratio = main['SE_lock'] / main['SE_full']
        obs = J['fracs']['1.0']['dstrat'] if D == 'D1b' else J['fracs']['1.0']['dpool']
        stab.append({'cand': K, 'learner': L or 'xgb', 'design': D, 'origin': J['v'], 'obs_delta_dev': obs, 'SE_full': se_f,
                     'SE_lock_scaled': se_f * ratio, 'power_lock_scaled': pw(tgt, se_f * ratio)})
res['stability'] = stab
# D3a 다중 원점(잠금 적용 불가, 정보)
d3a = []
for fn in sorted(glob.glob(C + r'\s3m_K*.json')):
    J = json.load(open(fn))
    d3a.append({'file': os.path.basename(fn), 'K': J['K'], 'L': J['L'], 'd_mean': J['d_mean'], 'se': J['se']})
res['D3a'] = d3a
# D3b 나라 풀링 투영(K1 ρ). Var_pool(m) ∝ V̄ (1 + (m−1)ρ)/m; 현재 m=2(CN·EP) 대비 비율
rho = json.load(open(C + r'\s3m_rho.json'))
rh = rho['rho_mean']
fac = lambda m, r: np.sqrt(((1 + (m - 1) * r) / m) / ((1 + r) / 2))
res['D3b'] = {'rho_mean': rh, 'rho_by_origin': {k: rho[k]['rho_k'] for k in rho if k.isdigit()},
              'factor_m4_main': float(fac(4, rh)), 'factor_m4_opt': float(fac(4, 0.0)), 'factor_m3_main': float(fac(3, rh))}
for r in [x for x in rows if x['design'] == 'D0']:
    for nm, f in [('D3b_main', fac(4, rh)), ('D3b_opt', fac(4, 0.0))]:
        se_l = r['SE_lock'] * f; se_i = r['SE_inf'] * f
        rows.append({**{k: r[k] for k in ['cand', 'learner', 'origin', 'obs_delta_dev', 'E_lock_pos', 'P0a', 'fL', 'fL_note']},
                     'design': nm, 'target': 0.002, 'SE_full': r['SE_full'] * f, 'SE_lock': se_l, 'SE_inf': se_i,
                     'power_lock': pw(0.002, se_l), 'MDE_lock': Z80 * se_l, 'MDE_inf': Z80 * se_i,
                     'G1': False, 'note': f'투영(×{f:.3f}, m=4 나라), 게이트 통과로 세지 않음(PREREG §3 D3)'})
res['rows'] = rows
json.dump(res, open(C + r'\s4_power.json', 'w'), indent=1, ensure_ascii=False, default=float)
cols = ['cand', 'learner', 'design', 'origin', 'target', 'obs_delta_dev', 'SE_full', 'SE_full_CGM', 'SE_full_NEST', 'SE_full_k', 'SE_full_strat', 'SE_full_f',
        'fit_a', 'fit_b', 'fL', 'fL_note', 'SE_lock', 'SE_inf', 'power_lock', 'MDE_lock', 'MDE_inf', 'need_lock_multiple_for_0.8', 'E_lock_pos', 'P0a', 'G1', 'note']
df = pd.DataFrame(rows)
for c in cols:
    if c not in df:
        df[c] = np.nan
df[cols].to_csv(C + r'\s4_power_rows.csv', index=False, encoding='utf-8-sig', float_format='%.6g')
for r in rows:
    out(r['cand'], r['learner'], r['design'], 'tgt', r['target'], 'obs %.5f' % r['obs_delta_dev'], 'SEfull %.5f' % r['SE_full'], 'SElock %.5f' % r['SE_lock'],
        'SEinf %.5f' % r['SE_inf'], 'pow %.3f' % r['power_lock'], 'MDE %.5f' % r['MDE_lock'], 'MDEinf %.5f' % r['MDE_inf'], 'G1', r['G1'])
out(json.dumps(res['D3b']))
for s in stab:
    out('stab', s['cand'], s['learner'], s['design'], s['origin'], 'obs %.5f' % s['obs_delta_dev'], 'SEf %.5f' % s['SE_full'], 'pow %.3f' % s['power_lock_scaled'])
for d in d3a:
    out('D3a', json.dumps(d))

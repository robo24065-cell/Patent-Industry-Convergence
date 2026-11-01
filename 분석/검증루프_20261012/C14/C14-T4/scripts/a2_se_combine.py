# C14-T4 a2: a1 부트스트랩 → 구조 확인 SE·검정력(PREREG §1.4–1.6), P0(c-i) 포함 관계 SE. 출력 se_struct.json(작업 폴더 최상위)
import sys, os, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
from scipy.stats import norm
a0 = json.load(open(C + r'\a0_out.json', encoding='utf-8'))
K = {k: np.load(C + rf'\a1_{k}.npz') for k in ['firm', 'sc', 'cell', 'prod']}
M = list(K['firm']['models']); ix = {m: i for i, m in enumerate(M)}
npos = K['firm']['npos']; pt = K['firm']['pt']
for k in K:
    assert np.allclose(K[k]['pt'], pt)


def stat_dA(A):   # A[..., v, m] → ΔAUC prov − max(free, hsc)
    return A[..., ix['prov']] - np.maximum(A[..., ix['free']], A[..., ix['hsc']])


def stat_dS(A):   # Δskill min (C12 d_v)
    p, f, h = A[..., ix['prov']], A[..., ix['free']], A[..., ix['hsc']]
    return np.minimum((p - f) / (1 - f), (p - h) / (1 - h))


def stat_E1(A):   # E_prov→pub = Δskill(all, pubctx)
    a, q = A[..., ix['all']], A[..., ix['pubctx']]
    return (a - q) / (1 - q)


def stat_E2(A):   # E_pub→prov = Δskill(all, prov)
    a, q = A[..., ix['all']], A[..., ix['prov']]
    return (a - q) / (1 - q)


def var_paths(fn, mean=False):
    V = {}
    for k in K:
        s = fn(K[k]['auc'])            # [b, v]
        if mean:
            s = s.mean(axis=1, keepdims=True)
        V[k] = np.var(s, axis=0, ddof=1)
    Vcgm = V['firm'] + V['sc'] - V['cell']
    Vpig = V['prod'] - 2 * V['cell']
    prim = np.maximum(np.maximum(Vcgm, V['firm']), V['sc'])
    r = {'SE_firm': np.sqrt(V['firm']), 'SE_sc': np.sqrt(V['sc']), 'SE_cell': np.sqrt(V['cell']), 'SE_prod': np.sqrt(V['prod']),
         'SE_CGM': np.sqrt(np.maximum(Vcgm, 0)), 'SE_pig': np.sqrt(np.maximum(Vpig, 0)), 'SE_primary': np.sqrt(prim),
         'cgm_pig_reldiff': np.abs(Vcgm - Vpig) / np.where(Vcgm > 0, Vcgm, np.nan)}
    return {k: [float(x) for x in np.atleast_1d(v)] for k, v in r.items()}


res = {'note': 'PREREG §1.4–1.6. 통계량 Δ = AUC(prov) − max(AUC(free), AUC(hsc)), 단일 원점. 구조 = a0 진단.',
       'structure': a0['struct'], 'repro_c12': a0['repro'],
       'clusters': {'firm': int(K['firm']['nf']), 'sc': int(K['firm']['ns']), 'cell': int(K['firm']['nc'])}, 'B': 400}
dA_pt = stat_dA(pt); dS_pt = stat_dS(pt)
res['point'] = {'dAUC_v': [float(x) for x in dA_pt], 'dAUC_mean': float(dA_pt.mean()), 'D_dskill_v': [float(x) for x in dS_pt],
                'D_mean': float(dS_pt.mean()), 'npos_v': [int(x) for x in npos],
                'E_prov_to_pub_mean': float(stat_E1(pt).mean()), 'E_pub_to_prov_mean': float(stat_E2(pt).mean()),
                'E_prov_to_pub_v': [float(x) for x in stat_E1(pt)], 'E_pub_to_prov_v': [float(x) for x in stat_E2(pt)]}
res['dAUC_single'] = var_paths(stat_dA)
res['dAUC_mean5'] = var_paths(stat_dA, mean=True)
res['D_dskill_mean5'] = var_paths(stat_dS, mean=True)
res['D_dskill_single'] = var_paths(stat_dS)
res['E_prov_to_pub_mean5'] = var_paths(stat_E1, mean=True)
res['E_pub_to_prov_mean5'] = var_paths(stat_E2, mean=True)
scale = np.sqrt(npos / E_LOCK_A)


def power(se_list, target=TARGET, sc=scale):
    se = np.array(se_list) * sc
    return [float(x) for x in norm.cdf(target / se - 1.96)]


S = res['dAUC_single']
pw = {k: power(S[k]) for k in ['SE_primary', 'SE_CGM', 'SE_pig', 'SE_prod', 'SE_firm', 'SE_sc']}
res['power_single_lock'] = {'E_lock': E_LOCK_A, 'scale_v': [float(x) for x in scale], 'target_dAUC': TARGET, **pw,
                            'SE_lock_primary_v': [float(x) for x in np.array(S['SE_primary']) * scale]}
res['P0b'] = {'min_power_primary': float(min(pw['SE_primary'])), 'pass': bool(min(pw['SE_primary']) >= 0.80),
              'min_power_prod_original_method': float(min(pw['SE_prod']))}
# 보고만: 관측 ΔAUC 평균에서의 검정력, 5원점 평균 기준
obs = float(dA_pt.mean())
res['offverdict'] = {'power_at_observed_mean_dAUC_primary': power(S['SE_primary'], target=obs),
                     'observed_mean_dAUC': obs,
                     'power_mean5_primary_noscale': float(norm.cdf(TARGET / res['dAUC_mean5']['SE_primary'][0] - 1.96)),
                     'dskill_equiv_of_target_v': [float(TARGET / (1 - max(p[ix['free']], p[ix['hsc']]))) for p in pt],
                     'C12_original_P0b_target_dskill_0.005_power_with_D_SE_primary': float(norm.cdf(0.005 / res['D_dskill_mean5']['SE_primary'][0] - 1.96)),
                     'C12_original_P0b_target_dskill_0.005_power_with_D_SE_prod': float(norm.cdf(0.005 / res['D_dskill_mean5']['SE_prod'][0] - 1.96))}
json.dump(res, open(W + r'\se_struct.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
out(json.dumps({k: res[k] for k in ['point', 'dAUC_single', 'D_dskill_mean5', 'power_single_lock', 'P0b', 'offverdict']}, ensure_ascii=False, indent=1))

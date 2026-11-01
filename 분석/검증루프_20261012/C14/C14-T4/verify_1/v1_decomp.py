# C14-T4 verify_1: P0(b) 원인 비중 반박 검정. 원천: ..\se_struct.json, ..\..\..\C12\C12-T4\gates.json (읽기만)
import json, math, sys
from scipy.stats import norm
base = sys.argv[1]
S = json.load(open(base + r'\C14\C14-T4\se_struct.json', encoding='utf-8'))
G = json.load(open(base + r'\C12\C12-T4\gates.json', encoding='utf-8'))
pw = lambda t, s: float(norm.cdf(t / s - 1.96))
se_c12 = G['P0']['b']['SE']['two']; t_c12 = 0.005
se_cgm = S['D_dskill_mean5']['SE_CGM'][0]; se_prod = S['D_dskill_mean5']['SE_prod'][0]
t_new = sum(S['offverdict']['dskill_equiv_of_target_v']) / 5
lock = S['power_single_lock']
r = {
 'orig_C12': pw(t_c12, se_c12),
 'SE_fix_only_CGM': pw(t_c12, se_cgm), 'SE_fix_only_prod': pw(t_c12, se_prod),
 'target_fix_only_origSE': pw(t_new, se_c12),
 'both': pw(t_new, se_cgm),
 'ratio_SE_c12_over_CGM': se_c12 / se_cgm, 'ratio_SE_c12_over_prod': se_c12 / se_prod, 'ratio_target': t_new / t_c12,
 'log_share_target_CGM': math.log(t_new / t_c12) / math.log((t_new / se_cgm) / (t_c12 / se_c12)),
 'log_share_target_prod': math.log(t_new / t_c12) / math.log((t_new / se_prod) / (t_c12 / se_c12)),
 'lock_power_recalc_primary': [pw(0.002, s) for s in lock['SE_lock_primary_v']],
 'lock_power_at_obs_mean': [pw(S['offverdict']['observed_mean_dAUC'], s) for s in lock['SE_lock_primary_v']],
}
json.dump(r, open('v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False, indent=1))

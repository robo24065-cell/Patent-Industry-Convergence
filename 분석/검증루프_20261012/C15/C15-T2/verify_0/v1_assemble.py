import json, sys, numpy as np, pandas as pd
D = sys.argv[1]
def load(prefix):
    P, R = {}, {}
    for f in [f'{prefix}_0_4', f'{prefix}_5_9']:
        j = json.load(open(f'{D}/verify_0/v1_t1_{f}.json')); P.update(j['point']); R.update({k: np.array(v) for k, v in j['reps'].items()})
    return P, R
Pm, Rm = load('m0'); Ps, Rs = load('sbb')
SB = pd.read_csv(f'{D}/seed_bundles.csv', encoding='utf-8-sig'); SB = SB[SB.track == 'T1'].set_index('bundle')
off = ["('off', 'logit_stack_y')", "('off', 'logit_single_y')", "('off', 'logit_stack_y_u')", "('off', 'logit_single_y_u')"]
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
rows = []
for b in range(10):
    k = lambda c: f"({b}, '{c}')"
    bl = off + [k('xgb_y'), k('xgb_y_u')]
    m0 = max(Pm[c] for c in bl); a = Pm[k('s')]; sk = (a - m0) / (1 - m0)
    bmax = np.max(np.column_stack([Rm[c] for c in bl]), 1); skb = (Rm[k('s')] - bmax) / (1 - bmax); ci = q(skb)
    asb = Ps[k('sbb')]; sks = (Ps[k('s')] - asb) / (1 - asb); sr = (Rs[k('s')] - Rs[k('sbb')]) / (1 - Rs[k('sbb')]); cis = q(sr)
    r = SB.loc[b]
    rows.append(dict(b=b, skill=sk, ci_lo=ci[0], ci_hi=ci[1], sbb_skill=sks, sbb_lo=cis[0], sbb_hi=cis[1],
                     d_skill=sk - r.skill, d_lo=ci[0] - r.ci_lo, d_hi=ci[1] - r.ci_hi, d_sbb=sks - r.skill_vs_strong, d_sbb_lo=cis[0] - r.strong_ci_lo, d_sbb_hi=cis[1] - r.strong_ci_hi))
o = pd.DataFrame(rows); pd.set_option('display.width', 250); print(o.round(6).to_string())
print('max abs diff', o.filter(like='d_').abs().max().max())
print('ranges: skill', o.skill.min(), o.skill.max(), 'lo', o.ci_lo.min(), o.ci_lo.max(), 'sbb_lo', o.sbb_lo.min(), o.sbb_lo.max(), 'n lo>0', (o.ci_lo > 0).sum(), (o.sbb_lo > 0).sum())
o.to_csv(f'{D}/verify_0/v1_t1_recomputed.csv', index=False)

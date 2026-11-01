# C09-T4 s8: rescore.csv 조립(T4 F22·개발 4원점, N4 final·개발 5원점) + PREREG §4 판정 계산.
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
OFF = {'T4': (0.246, 0.133, 0.320), 'N4': (0.177, 0.147, 0.209)}
rows = []
for o in ['F22', 'O18', 'O16', 'O14', 'O12']:
    f = os.path.join(C, f't4_score_{o}.json')
    if not os.path.exists(f): continue
    r = json.load(open(f, encoding='utf-8'))
    imp = {}
    for rule in ['official', 'R1', 'R2', 'R3']:
        if f'skill_{rule}' not in r: continue
        row = dict(track='T4', window='final' if o == 'F22' else 'dev', origin=o, rule='orig' if rule == 'official' else rule, n=r['n'],
                   auc=r[f'auc_{rule}'], auc_base=r['auc_base'], skill=r[f'skill_{rule}'], skill_ci_lo=r[f'skill_ci_{rule}'][0], skill_ci_hi=r[f'skill_ci_{rule}'][1],
                   delta=r.get(f'delta_{rule}', 0.0), delta_ci_lo=r.get(f'delta_ci_{rule}', [0, 0])[0], delta_ci_hi=r.get(f'delta_ci_{rule}', [0, 0])[1], boot_B=r['B'])
        if o == 'F22':
            fi = os.path.join(C, f"t4_importance_{'orig' if rule == 'official' else rule}.json")
            if os.path.exists(fi):
                im = json.load(open(fi, encoding='utf-8'))
                row['late_gain_best_rank'] = min(v for v in im['gain_rank'].values() if v is not None)
                row['late_gain_ranks'] = json.dumps(im['gain_rank'])
                row['late_perm_auc_drop'] = im['perm_auc_drop']
        rows.append(row)
for mode in ['final', 'dev']:
    f = os.path.join(C, f'n4_rescore_{mode}.json')
    for r in json.load(open(f, encoding='utf-8')):
        for rule in ['orig', 'R1', 'R2', 'R3']:
            im = r['importance'][rule]
            rows.append(dict(track='N4', window=mode, origin=str(r['t0']), rule=rule, n=r['n'], auc=r['auc'][rule], auc_base=r['auc_M0'], skill=r['skill'][rule],
                             skill_ci_lo=r[f'skill_ci_{rule}'][0], skill_ci_hi=r[f'skill_ci_{rule}'][1],
                             delta=r['skill'][rule] - r['skill']['orig'], delta_ci_lo=r.get(f'delta_ci_{rule}', [0, 0])[0], delta_ci_hi=r.get(f'delta_ci_{rule}', [0, 0])[1],
                             boot_B=r['n_boot'], late_gain_best_rank=min(im['gain_rank'].values()), late_gain_ranks=json.dumps(im['gain_rank']), late_perm_auc_drop=im['perm_auc_drop']))
d = pd.DataFrame(rows)
# 판정(최종창)
ver = {}
for tr in ['T4', 'N4']:
    s0, lo, hi = OFF[tr]
    fin = d[(d.track == tr) & (d.window == 'final')]
    caveat = []
    for _, x in fin[fin.rule != 'orig'].iterrows():
        a, b = s0 + x.delta_ci_lo, s0 + x.delta_ci_hi
        if a < lo or b > hi: caveat.append(x.rule)
    o0 = fin[fin.rule == 'orig'].iloc[0]
    top = (o0.late_gain_best_rank <= 10) or (o0.late_perm_auc_drop >= 0.005)
    gone = []
    if top:
        for _, x in fin[fin.rule != 'orig'].iterrows():
            if (x.late_gain_best_rank > 10) and (x.late_perm_auc_drop < 0.5 * o0.late_perm_auc_drop): gone.append(x.rule)
    dev = d[(d.track == tr) & (d.window == 'dev') & (d.rule != 'orig')]
    devsum = {r: dict(mean_delta=float(g.delta.mean()), n_origins=int(len(g)), n_neg=int((g.delta < 0).sum())) for r, g in dev.groupby('rule')}
    ver[tr] = dict(caveat_rules=caveat, late_top_contribution_before=bool(top), disappeared_rules=gone,
                   verdict=('단서 필요' if caveat else ('해석 수정 필요' if gone else '영향 없음')),
                   interp_criterion=('해당 없음(보정 전 상위 기여 아님)' if not top else ('사라짐: ' + ','.join(gone) if gone else '유지')),
                   official_plus_delta_ci={x.rule: [round(s0 + x.delta_ci_lo, 4), round(s0 + x.delta_ci_hi, 4)] for _, x in fin[fin.rule != 'orig'].iterrows()},
                   dev_summary=devsum)
d['verdict_track'] = d.track.map({k: v['verdict'] for k, v in ver.items()})
d.to_csv(os.path.join(W, 'rescore.csv'), index=False, encoding='utf-8-sig')
json.dump(ver, open(os.path.join(C, 's8_verdict.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(d[['track', 'window', 'origin', 'rule', 'auc', 'skill', 'delta', 'delta_ci_lo', 'delta_ci_hi']].round(4).to_string())
print(json.dumps(ver, ensure_ascii=False, indent=1))

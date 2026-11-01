# -*- coding: utf-8 -*-
"""정리: b_sens.csv(시나리오×seed, 짝 Δ), a_concentration.csv(정의별 SAB), 판정 계산. 입력은 이번 바퀴 cache·verify_0 산출물."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
R = {}
for fp in glob.glob(os.path.join(CACHE, 'b3_*.json')):
    R.update(json.load(open(fp, encoding='utf-8')))
rows = [dict(scenario=k.split('|')[0], seed=int(k.split('|')[1]), **{kk: v for kk, v in r.items() if kk != 'nan_rate'},
             nan_train_D_g2=r['nan_rate']['pi_prod_g2']['train_D'], nan_train_D_g1=r['nan_rate']['pi_prod_g1']['train_D'],
             nan_train_W_g2=r['nan_rate']['pi_prod_g2']['train_W']) for k, r in R.items()]
B = pd.DataFrame(rows).sort_values(['scenario', 'seed'])
s0 = B[B.scenario == 'S0'].set_index('seed').skill
B['d_skill_vs_S0'] = B.skill - B.seed.map(s0)
summ = []
for sc, g in B.groupby('scenario'):
    d = g.d_skill_vs_S0.values; m = float(d.mean()); sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    if sc == 'S0': v = '기준'
    elif abs(m) >= 0.0268: v = '영향 있음'
    elif abs(m) > 0.0044: v = '작은 영향(경고)'
    else: v = '영향 없음'
    summ.append(dict(scenario=sc, seed='mean(0-4)', skill=float(g.skill.mean()), auc=float(g.auc.mean()), d_skill_vs_S0=m, d_sd=sd,
                     n_pos=int((d > 0).sum()), hits_top10=float(g.hits_top10.mean()), spearman_vs_official=float(g.spearman_vs_official.mean()),
                     verdict=v + (' (판정 시나리오)' if sc in ('S1', 'S2') else (' (보고용)' if sc != 'S0' else ''))))
S = pd.DataFrame(summ)
out = pd.concat([B, S], ignore_index=True)
cols = ['scenario', 'seed', 'auc', 'auc_base', 'skill', 'd_skill_vs_S0', 'd_sd', 'n_pos', 'hits_top10', 'spearman_vs_official', 'maxdiff_vs_official', 'yb_agree',
        'nan_train_D_g2', 'nan_train_D_g1', 'nan_train_W_g2', 'verdict']
out[cols].to_csv(os.path.join(WD, 'b_sens.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(S.round(5).to_string())
# verify V-B 대조
vb = json.load(open(os.path.join(WD, 'verify_0', 'vB_out.json'), encoding='utf-8'))
print('VB S1 diff', abs(vb['S1_seed0'] - R['S1|0']['skill']), abs(vb['S1_seed1'] - R['S1|1']['skill']))
for v in ['V1', 'V2']:
    print(v, [round(vb[f'{v}_seed{s}'] - R[f'S0|{s}']['skill'], 5) for s in (0, 1)])
# a_concentration.csv
A = json.load(open(os.path.join(CACHE, 'a2_results.json'), encoding='utf-8'))
va = json.load(open(os.path.join(WD, 'verify_0', 'vA_out.json'), encoding='utf-8'))
role = {'i_name': '원판(재현)', 'ii_code': '주 판정', 'iii_alljoint': '주 판정', 'iv_group_on_iii': '경고용', 'ivb_C04G_on_name': '보고(C04 G 재현)', 'iv_group_on_ii': '보고',
        'P_impute': '경고용', 'R_reg_name': '한계(등록 부분집합)', 'R_reg_code': '한계(등록 부분집합)', 'R_reg_alljoint': '한계(등록 부분집합)', 'H_upper': '한계(최악 상한)', 'H_lower': '한계(최악 하한)'}
ar = []
for t, r in A['results'].items():
    ar.append(dict(definition=t, role=role.get(t, ''), note=r['note'], n_sab=r['n_sab'], rate=r['rate'], d_rate_pp=(r['rate'] - 205 / 504) * 100, skill_excl=r['skill_excl'],
                   d_skill_excl=r['skill_excl'] - 0.4378, skill_B=r['skill_B'], rate_S0=r['rate_by_stratum']['S0'], rate_S1=r['rate_by_stratum']['S1'], rate_S2=r['rate_by_stratum']['S2'],
                   rate_D5=r['rate_D5'], hits_top10_sab=r['hits_top10_sab'], flips_out=A['flips_vs_i'].get(t, {}).get('out_', 0), flips_in=A['flips_vs_i'].get(t, {}).get('in_', 0),
                   verdict=(A['verdict'][t] if role.get(t) in ('원판(재현)', '주 판정') else
                            ('경고(정의 의존)' if A['verdict'][t] != '영향 없음' and t in ('iv_group_on_iii', 'P_impute', 'ivb_C04G_on_name', 'iv_group_on_ii') else
                             ('한계(등록 부분집합: D가 바뀌어 원판과 직접 비교 불가; 같은 집합 안 i→iii 차 +0.8%p·+0.0009)' if t.startswith('R_') else
                              ('한계(미등록 불확정: 범위 [36.1%, 97.8%]가 문턱을 걸침)' if t.startswith('H_') else A['verdict'][t]))))))
M = A['M_mc']
ar.append(dict(definition='M_montecarlo', role='보고(미등록 대치 200회)', note='(iii)+미등록 문서 공동출원 확률 대치, 평균 [2.5%, 97.5%]', n_sab=M['n_sab']['mean'], rate=M['rate']['mean'],
               d_rate_pp=(M['rate']['mean'] - 205 / 504) * 100, skill_excl=M['skill_excl']['mean'], d_skill_excl=M['skill_excl']['mean'] - 0.4378,
               verdict=f"rate [{M['rate']['q025']:.4f}, {M['rate']['q975']:.4f}] skill_excl [{M['skill_excl']['q025']:.4f}, {M['skill_excl']['q975']:.4f}]"))
for k in ['i_>', 'ii_k1_>', 'iii_k1_>']:
    r = va[k]; ar.append(dict(definition='boundary_strict_' + k.split('_')[0], role='반박(경계: Δa > 0.5·D)', note='verify_0 SQL, 등호 제외(과반)', n_sab=r['n_sab'], rate=r['rate'],
                              d_rate_pp=(r['rate'] - 205 / 504) * 100, skill_excl=r['skill_excl'], d_skill_excl=r['skill_excl'] - 0.4378, verdict='정의 민감도(보고)'))
pd.DataFrame(ar).to_csv(os.path.join(WD, 'a_concentration.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
print(pd.DataFrame(ar)[['definition', 'n_sab', 'rate', 'd_rate_pp', 'skill_excl', 'd_skill_excl', 'verdict']].round(4).to_string())

# -*- coding: utf-8 -*-
"""B단계 종료 시 현재 챔피언(champion.json)의 spec 을 final_spec_r3.json 으로 동결한다(1회)."""
import os, json
import pandas as pd
R3 = os.path.dirname(os.path.abspath(__file__))
ch = json.load(open(os.path.join(R3, 'champion.json')))['exp']
S = json.load(open(os.path.join(R3, 'specs', f'{ch}.json'), encoding='utf-8'))
exp = pd.read_csv(os.path.join(R3, 'experiments.csv'))
acc = exp[exp.decision.astype(str).str.startswith('ACCEPT')]
row = exp[exp.exp_id == ch].iloc[0]
chain = ['E00: 2라운드 동결 모형(R2 E03) 재채점, 평균 skill %.4f' % exp[exp.exp_id == 'E00'].mean_skill.iloc[0]]
for _, r in acc.iterrows():
    chain.append(f'{r.exp_id} (parent {r.parent}): {r.change} — ACCEPT {r.diff_vs_champ:+.4f}, 우세 {r.win_rate:.2f}, 평균 skill {r.mean_skill:.4f}')
out = dict(
    track='T4 특허 포기(등록료불납) 예측 — 3라운드',
    version=f'R3-{ch}',
    source_exp=ch,
    frozen_at='2026-09-29 (3라운드 B단계 종료)',
    change_chain=chain,
    dev_origins={'O12': ['2012-01-01', ['2012-01-01', '2012-07-01']], 'O14': ['2014-01-01', ['2014-01-01', '2014-07-01']],
                 'O16': ['2016-01-01', ['2016-01-01', '2016-07-01']], 'O18': ['2018-01-01', ['2018-01-01', '2018-07-01']]},
    dev_mean_skill=float(row.mean_skill),
    dev_per_origin={o: dict(auc=float(row[f'{o}_auc']), auc_base=float(row[f'{o}_base']), skill=float(row[f'{o}_skill'])) for o in ('O12', 'O14', 'O16', 'O18')},
    final_test='없음(PROTOCOL_R3 §3). 공식 성적은 2라운드 최종시험 skill 0.246 [0.133, 0.320] 유지',
    future=dict(C='2025-08-22', trend_L='2012-01-01..2025-07-01 반기', output='R3/future_forecast_r3.csv'),
    selection_rule='PROTOCOL_R2 §3 / R3 §4: 개발 원점 평균 skill +0.003 이상 AND 원점 60% 이상 우세',
    reuse_r2_x=S.get('reuse_r2_x', []),
    spec=S['spec'],
)
json.dump(out, open(os.path.join(R3, 'final_spec_r3.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('동결', ch, row.mean_skill)

# -*- coding: utf-8 -*-
r"""C19-T2 s8: 채점 결과 모아 sens_results.csv 와 사전 판정(cache/s8_verdict.json)."""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd

off = json.load(open(J(T4, 'R2', 'final_result.json'), encoding='utf-8'))
ci = off['bootstrap_95ci']['skill']; THR = round(0.25 * (ci[1] - ci[0]), 4)
S = {v: json.load(open(J(C, f'score_{v}.json'), encoding='utf-8')) for v in ['OFFICIAL', 'P0', 'F1', 'F1x', 'F2', 'F3s1', 'F3s2'] if os.path.exists(J(C, f'score_{v}.json'))}
desc = {'OFFICIAL': '공식 저장 예측(같은 채점 경로)', 'P0': '재현: 원문 재계산 fam_n 으로 재학습', 'F1': 'fam_n 정규화(국가+접미+앞0 제거, 과제 지정 규칙)',
        'F1x': 'fam_n 확장 정규화(F1 + US 공개 11↔10자리, JP 공개 연호↔서기) [보조]', 'F2': 'fam_n 제거(상수 = frame_cls 평균)',
        'F3s1': '위약 seed 1(연도별 실제 중복률로 무작위 토큰 ±)', 'F3s2': '위약 seed 2'}
rows = []
for v, r in S.items():
    rows.append(dict(variant=v, desc=desc[v], auc=round(r['auc'], 6), auc_M0=round(r['auc_M0'], 6), skill=round(r['skill'], 6), d_skill=round(r['d_skill'], 6),
                     d_skill_ci_lo=round(r['d_skill_ci'][0], 6), d_skill_ci_hi=round(r['d_skill_ci'][1], 6), d_auc=round(r['d_auc'], 6),
                     ci_M0_lo=round(r['ci_M0'][0], 4), ci_M0_hi=round(r['ci_M0'][1], 4), class_M0=r['class_M0'],
                     skill_vs_strong=round(r['skill_vs_strong'], 4), ci_strong_lo=round(r['ci_strong'][0], 4), ci_strong_hi=round(r['ci_strong'][1], 4), class_strong=r['class_strong'],
                     top1000_hits=r['top1000_hits'], top1pct_hits=r['top1pct_hits'], hit100=r['hit100'], top1000_overlap_official=r['top1000_overlap_official'],
                     comps_auc=json.dumps({k: round(x, 5) for k, x in r['comps_auc'].items()}),
                     max_comp_diff_vs_official=json.dumps(r.get('max_comp_diff_vs_official', {})), seed_or_rule=''))
df = pd.DataFrame(rows)
V = {'threshold_0.25xCIwidth': THR, 'official_ci': ci}
if 'F3s1' in S and 'F3s2' in S:
    pl = [min(S['F3s1']['d_skill_ci'][0], S['F3s2']['d_skill_ci'][0]), max(S['F3s1']['d_skill_ci'][1], S['F3s2']['d_skill_ci'][1])]
    V['placebo_range'] = pl
    for v in ['F1', 'F1x']:
        if v not in S:
            continue
        r = S[v]; d = r['d_skill']
        c1 = abs(d) < THR; c2 = pl[0] <= d <= pl[1]; c3 = (r['class_M0'] == S['OFFICIAL']['class_M0']) and (r['class_strong'] == S['OFFICIAL']['class_strong'])
        verdict = '영향 없음' if (c1 and c2 and c3) else '경고'
        err = (r['skill'] < S['OFFICIAL']['skill']) and abs(d) >= THR
        V[v] = dict(d_skill=d, d_skill_ci=r['d_skill_ci'], abs_lt_thr=c1, in_placebo=c2, class_same=c3, verdict=verdict + (' [보조]' if v == 'F1x' else ''), manuscript_error_candidate=err)
if 'P0' in S:
    p = S['P0']
    V['P0'] = dict(max_comp_diff=p['max_comp_diff_vs_official'], max_ens_diff=p['max_ens_diff_vs_official'], d_auc=p['d_auc'], d_skill=p['d_skill'],
                   ci_M0_4dp=p['ci_M0_4dp'], official_ci=ci, pass_=(max(p['max_comp_diff_vs_official'].values()) <= 1e-6 and abs(p['d_skill']) <= 1e-6 and p['ci_M0_4dp'] == ci))
df['verdict'] = df.variant.map(lambda v: V.get(v, {}).get('verdict', ''))
df.to_csv(J(W, 'sens_results.csv'), index=False, encoding='utf-8-sig')
jdump(V, J(C, 's8_verdict.json'))
out(json.dumps(V, ensure_ascii=False, default=float))
print(df[['variant', 'auc', 'skill', 'd_skill', 'd_skill_ci_lo', 'd_skill_ci_hi', 'class_M0', 'skill_vs_strong', 'ci_strong_lo', 'class_strong', 'top1000_hits', 'top1pct_hits', 'hit100']].to_string())

# -*- coding: utf-8 -*-
"""variants.csv 조립 + verdicts.json(PREREG §5 판정)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16common import *
OFF = json.load(open(CC/'var'/'OFF.json', encoding='utf-8')); TH = 0.0087
DESC = {'OFF': ('공식', '-', '공식'), 'S1full_La': ('S1 완전 가시화(주, C 법인 행 재계산)', 'v_pub', 'L-a'), 'S1full_Lb': ('S1 완전 가시화(C 법인 행 재계산)', 'v_pub', 'L-b'),
        'S1fixed_La': ('S1 완전 가시화(공식 행 공식 점수)', 'v_pub', 'L-a'), 'S1fixed_Lb': ('S1 완전 가시화(공식 행 공식 점수)', 'v_pub', 'L-b'),
        'S2_v_pub': ('S2 ≥3건만 가시화', 'v_pub', '공식'), 'S2_v_p18': ('S2 ≥3건만 가시화', 'v_p18', '공식'), 'S2_v_rgst': ('S2 ≥3건만 가시화', 'v_rgst', '공식'),
        'S3_La': ('S3 prior_any만 가시화', 'app v', 'L-a'), 'S3_Lb': ('S3 prior_any만 가시화', 'app v', 'L-b'),
        'ALIAS_i_exclude': ('별칭 민감도 i: 오매칭 의심 P 양성 제외', '-', '공식'), 'ALIAS_ii_relabel0': ('별칭 민감도 ii: 의심 P 양성을 음성으로', '-', '공식'),
        'ALIAS_iii_addR': ('별칭 민감도 iii: 오매칭 R 쌍 되돌림', '-', '공식'), 'ALIAS_iv_ii_iii': ('별칭 민감도 ii+iii', '-', '공식')}
rows = []
for nm, (d, v, l) in DESC.items():
    r = json.load(open(CC/'var'/f'{nm}.json', encoding='utf-8')); inf = r.get('info', {})
    dl = r['skill'] - OFF['skill']
    cls_same = (r['ci'][0] > 0) == (OFF['ci'][0] > 0) and (r['ci_vs_sbb'][0] > 0) == (OFF['ci_vs_sbb'][0] > 0)
    verdict = '공식' if nm == 'OFF' else ('경고(분류 변경)' if not cls_same else ('경고(Δ≤−0.0087)' if dl <= -TH else ('영향 있음(+, 공식이 보수적)' if dl >= TH else '영향 없음')))
    rows.append(dict(variant=nm, desc=d, vis=v, label=l, n_pairs=r['n'], firms=r['firms'], pos=r['pos'], base_rate=round(r['base_rate'], 6), auc=round(r['auc'], 6), auc_M0=round(r['auc_M0'], 6), M0_arg=r['M0_arg'],
                     skill=round(r['skill'], 5), ci_lo=round(r['ci'][0], 5), ci_hi=round(r['ci'][2], 5), delta_skill=round(dl, 5), skill_vs_sbb=round(r['skill_vs_sbb'], 5), sbb_ci_lo=round(r['ci_vs_sbb'][0], 5), sbb_ci_hi=round(r['ci_vs_sbb'][2], 5),
                     skill_new=round(r['skill_new'], 4), top100=r['top100'], top1000=r['top1000'], capture_top1pct=round(r['capture_top1pct'], 4), top1000_C=r.get('top1000_C', 0),
                     D_dropped=inf.get('D_dropped', ''), D_pos=inf.get('D_dropped_pos', ''), B_added=inf.get('B_added', ''), B_officialdef_pos=inf.get('B_officialdef_pos', ''),
                     C_added=inf.get('C_added', ''), C_officialdef_pos=inf.get('C_officialdef_pos', ''), verdict=verdict))
V0 = json.load(open(HERE/'verify_0'/'v0b_out.json', encoding='utf-8'))
for nm, d in (('S2_indep', 'verify_0 독립(등록원부 출원일·DB 직접) S2'), ('S1fixed_La_indep', 'verify_0 독립 S1-fixed·L-a(C 1,798)'), ('S1full_La_indep', 'verify_0 독립 S1-full·L-a(C 1,798)'),
              ('r1_vrgst_worst', '반박 r1 v_rgst, 점수 없는 C 최악'), ('r1_vrgst_best', '반박 r1 v_rgst, 점수 없는 C 최선'), ('r2_vpub_worst', '반박 r2 C 전부 최악'), ('r2_vpub_best', '반박 r2 C 전부 최선')):
    r = V0[nm]; rows.append(dict(variant=nm, desc=d, vis='registry' if 'indep' in nm else ('v_rgst' if 'r1' in nm else 'registry'), label='L-a' if 'La' in nm or 'r' in nm[:2] else '공식',
                                 n_pairs=r.get('n'), pos=r.get('pos'), skill=round(r['skill'], 5), delta_skill=round(r['delta'], 5), C_added=r.get('C_scored', r.get('C_extreme', '')),
                                 verdict='CI 없음(점추정·경계)'))
pd.DataFrame(rows).to_csv(HERE/'variants.csv', index=False, encoding='utf-8-sig')
s1 = json.load(open(CC/'var'/'S1full_La.json', encoding='utf-8'))
share = (s1['info']['D_dropped'] + s1['info']['B_added'] + s1['info']['C_added']) / OFF['n']
ver = dict(main=dict(variant='S1full_La', delta=s1['skill'] - OFF['skill'], verdict=[r['verdict'] for r in rows if r['variant'] == 'S1full_La'][0],
                     M0_class=s1['M0_class'], SBB_class=s1['SBB_class'], paired_delta_ci=V0['r3_paired_delta']['ci'], indep_delta=V0['S1full_La_indep']['delta']),
           post_origin_dependent_share=share, disclosure='단서 누락 후보' if share > 0 else '해당 없음',
           alias=json.load(open(CC/'alias_summary.json', encoding='utf-8'))['sample_final'])
jdump(ver, HERE/'verdicts.json'); print(json.dumps(ver['main'], ensure_ascii=False), share)
print(pd.DataFrame(rows)[['variant', 'n_pairs', 'pos', 'skill', 'ci_lo', 'ci_hi', 'delta_skill', 'skill_vs_sbb', 'sbb_ci_lo', 'top1000', 'verdict']].to_string())

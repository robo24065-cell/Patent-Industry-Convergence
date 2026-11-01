# -*- coding: utf-8 -*-
"""cache/var/*.json → variants.csv, verdicts.json"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard17 as G
from c17common import *
VD = CC/'var'
DESC = {
 'OFF': ('공식(재현)', '-', '공식'),
 'N1': ('T1-02 N1 고정판: 원점 가시 증거(E_pub∪E_reg∪E_own) 없는 신규진입자 법인 제외', 'fixed', '보조'),
 'N1re': ('T1-02 N1 재계산판(주): 같은 법인 제외 후 전체 재구축', 'recompute', '주(T1-02)'),
 'N1nar': ('반박 r1: N1 증거 좁게(E_pub만)', 'fixed', '반박'),
 'N1rl': ('반박 r1: N1 증거 넓게(E_rl_ins, 취득일 미확인 포함)', 'fixed', '반박'),
 'N1wide': ('반박 r1: N1 증거 문언 최대(권리목록 범위밖 포함)', 'fixed', '반박'),
 'N2': ('T1-02 N2 고정판: 신규진입자 쌍 전부 제외', 'fixed', '보조'),
 'N2re': ('T1-02 N2 재계산판', 'recompute', '보조'),
 'R1fx': ('T1-05 R1 고정판: RC:001 법인 308곳 행 추가(공식 행 공식 점수)', 'fixed', '보조'),
 'R1re': ('T1-05 R1 재계산판(주): RC 법인 주입 전체 재구축', 'recompute', '주(T1-05)'),
 'R1re_worst': ('반박 r3: R1 RC 행 모형 점수 최악 대입', 'bound', '반박'),
 'R1re_best': ('반박 r3: R1 RC 행 모형 점수 최선 대입', 'bound', '반박'),
 'R1re_rand': ('사후 탐색(PREREG 밖): R1 RC 행 모형 점수를 공식 행 무작위 추출로', 'explore', '탐색'),
 'S1full': ('C16 S1-full·L-a 재현(대조)', 'recompute', '대조'),
 'S1fixed': ('C16 S1-fixed·L-a 재현(대조)', 'fixed', '대조'),
 'S1N1_re': ('S1-full + N1(S1판) 재계산판', 'recompute', '보조'),
 'S1N1_fx': ('S1-fixed + N1(S1판) 고정판', 'fixed', '보조'),
 'CMB_re': ('결합(주): S1-full + N1 + R1 재계산판', 'recompute', '주(결합)'),
 'CMB_fx': ('결합 고정판: S1-fixed + N1 + R1', 'fixed', '보조'),
}
for pool in ('neg', 'all'):
    for s in (1, 2, 3):
        DESC[f'N3_{pool}_s{s}'] = (f'T1-02 N3 선택 완성판: 공식 + 가짜 법인 1,304(공여 {pool}, 시드 {s})', 'donor', '보조' if pool == 'neg' else '반박 r2')
        DESC[f'N3r_{pool}_s{s}'] = (f'T1-02 N3r 대체판: N1 + 가짜 법인 1,304(공여 {pool}, 시드 {s})', 'donor', '보조' if pool == 'neg' else '반박 r2')
DESC['N3x3_neg_x3'] = ('반박 r2: 공식 + 가짜 법인 3,912(공여 neg, 시드 1)', 'donor', '반박 r2')
rows = []; R = {}
for nm, (desc, kind, role) in DESC.items():
    d = json.load(open(VD/f'{nm}.json', encoding='utf-8')); R[nm] = d
    dc = d.get('delta_ci', [None] * 3)
    rows.append(dict(variant=nm, role=role, kind=kind, desc=desc, n_pairs=d['n'], firms=d['firms'], pos=d['pos'], base_rate=round(d['base_rate'], 6), auc=round(d['auc'], 6), auc_M0=round(d['auc_M0'], 6), M0_arg=d['M0_arg'],
                     skill=round(d['skill'], 5), ci_lo=round(d['ci'][0], 5), ci_hi=round(d['ci'][2], 5), delta_skill=round(d.get('delta', 0.0), 5),
                     dci_lo=None if dc[0] is None else round(dc[0], 5), dci_hi=None if dc[2] is None else round(dc[2], 5), share_delta_le_m0087=d.get('delta_le_thr_share'),
                     skill_vs_sbb=round(d['skill_vs_sbb'], 5), sbb_ci_lo=round(d['ci_vs_sbb'][0], 5), skill_new=None if d['skill_new'] is None else round(d['skill_new'], 4),
                     top100=d['top100'], top1000=d['top1000'], capture_top1pct=round(d['capture_top1pct'], 4), added_rows=d.get('added_rows', 0), added_pos=d.get('added_pos', 0),
                     added_score_pct_mean=(d.get('added_score_pct') or {}).get('mean'), M0_class=d['M0_class'], SBB_class=d['SBB_class'], verdict=d.get('verdict', '공식'), info=json.dumps(d['info'], ensure_ascii=False)))
V = pd.DataFrame(rows); V.to_csv(HERE/'variants.csv', index=False, encoding='utf-8-sig')
rng = lambda names: [min(R[n]['delta'] for n in names), max(R[n]['delta'] for n in names)]
v0 = json.load(open(HERE/'verify_0'/'v0_out.json', encoding='utf-8'))
out = dict(
 official=dict(skill=R['OFF']['skill'], ci_paired_boot=R['OFF']['ci'], M0_class=R['OFF']['M0_class'], SBB_class=R['OFF']['SBB_class']),
 main={k: dict(variant=v, skill=R[v]['skill'], delta=R[v]['delta'], delta_ci=R[v]['delta_ci'], skill_ci=R[v]['ci'], M0_class=R[v]['M0_class'], SBB_class=R[v]['SBB_class'], verdict=R[v]['verdict'])
       for k, v in (('T1-02 단독', 'N1re'), ('T1-05 단독', 'R1re'), ('결합 S1-full+N1+R1', 'CMB_re'))},
 auxiliary=dict(N1_fixed=R['N1']['delta'], N2_fixed=R['N2']['delta'], N2_recompute=R['N2re']['delta'], R1_fixed=R['R1fx']['delta'], S1N1_re=R['S1N1_re']['delta'], S1N1_fx=R['S1N1_fx']['delta'], CMB_fx=R['CMB_fx']['delta'],
                N3_neg_range=rng([f'N3_neg_s{s}' for s in (1, 2, 3)]), N3_all_range=rng([f'N3_all_s{s}' for s in (1, 2, 3)]), N3r_neg_range=rng([f'N3r_neg_s{s}' for s in (1, 2, 3)]),
                N3r_all_range=rng([f'N3r_all_s{s}' for s in (1, 2, 3)]), N3x3=R['N3x3_neg_x3']['delta'], S1full_repro=R['S1full']['skill'], S1fixed_repro=R['S1fixed']['skill']),
 refutation=dict(
   r1_evidence_definition=dict(N1nar=R['N1nar']['delta'], N1=R['N1']['delta'], N1rl=R['N1rl']['delta'], N1wide=R['N1wide']['delta'], verdicts=sorted(set(R[n]['verdict'] for n in ('N1nar', 'N1', 'N1rl', 'N1wide'))),
                               result='뒤집히지 않음(네 정의 모두 영향 없음, Δ 부호 모두 +)'),
   r2_N3_pool_size=dict(verdicts=sorted(set(R[n]['verdict'] for n in R if n.startswith('N3'))), result='뒤집히지 않음(공여 풀·시드·3배 규모 모두 영향 없음; AUC는 최대 −0.018 하락하나 M0도 함께 하락)'),
   r3_R1_bounds=dict(worst=R['R1re_worst']['delta'], worst_verdict=R['R1re_worst']['verdict'], best=R['R1re_best']['delta'], best_verdict=R['R1re_best']['verdict'], rand_explore=R['R1re_rand']['delta'],
                     result='부분 반박: 최악 대입(RC 음성 14,795행을 모두 최상위)이면 경고(−0.052)·최선이면 +0.0087(문턱 경계). 실제 재구축 점수는 RC 행 평균 백분위 0.35라 최악 시나리오와 거리가 멀고, 무작위 점수(사후 탐색)에서도 +0.0005. R1 판정은 RC 법인 재구축 점수(범위밖 0건 가정)에 의존한다.')),
 verify_0=dict(N1_delta_indep=v0['N1_indep_def']['delta'], N1_delta_main=R['N1']['delta'], N2_delta_indep=v0['N2_official_newc']['delta'], N2_delta_main=R['N2']['delta'],
               evidence_sym_diff=v0['N1_vis_sym_diff'], newc_registry_applDate_def=v0['newc_indep_registry_applDate'], N2_delta_registry_def=v0['N2_indep_newc']['delta'],
               pass_=bool(abs(v0['N1_indep_def']['delta'] - R['N1']['delta']) <= 0.001 and abs(v0['N2_official_newc']['delta'] - R['N2']['delta']) <= 0.001)),
 manuscript=dict(sha1='39452163a4b95d0600525dc9b9c1b85d4da8fe80', lines=806, rule='주 판정 모두 영향 없음 → 정성 단서만 권장(PREREG §4). 410행 수치 범위만 0자 대체 제안(E-C17T1-01).'))
jdump(out, HERE/'verdicts.json'); print(V[['variant', 'role', 'skill', 'ci_lo', 'ci_hi', 'delta_skill', 'dci_lo', 'dci_hi', 'top1000', 'verdict']].to_string())

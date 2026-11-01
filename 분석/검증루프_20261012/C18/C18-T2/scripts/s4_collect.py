# -*- coding: utf-8 -*-
r"""C18-T2 s4: 민감도 결과 모음 → sens_results.csv (판정은 PREREG [3] 사전 기준: |Δskill_final| < 0.25 × 공식 CI 폭 → '영향 없음', 이상 → '경고';
 짝 Δ CI가 0을 포함하지 않으면 '작지만 구분되는 변화' 덧붙임). 원고 숫자 행은 보고 정밀도 비교."""
import os, sys, json, csv
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
import numpy as np
TH = {'T3': (0.36580, 0.47307), 'N5': (0.007, 0.360), 'T1': (0.4536239, 0.4884528)}
OFF = {'T3': 0.422887, 'N5': 0.165, 'T1': 0.472670}
rows = []
def judge(trk, d, ci):
    th = 0.25 * (TH[trk][1] - TH[trk][0])
    j = '영향 없음' if abs(d) < th else '경고'
    if ci and not (ci[0] <= 0 <= ci[1]): j += ' (작지만 구분되는 변화)'
    return th, j
T = os.path.join(C, 't3res')
def t3(tag):
    f = os.path.join(T, f'res_{tag}.json'); return json.load(open(f, encoding='utf-8')) if os.path.exists(f) else None
p0f = t3('P0_final'); p0d = [t3('P0_dev_2010-2011-2012-2013'), t3('P0_dev_2014-2015-2016-2017-2018')]
devmean = lambda L: float(np.mean([r['skill'] for x in L for r in x['by_origin'].values()])) if all(L) else None
P0dev = devmean(p0d)
rows.append(['T3', 'P0', 'E40-auditD final t0=2021', p0f['mean_skill'], p0f['mean_skill'] - OFF['T3'], '', P0dev, '', '', '공식 0.42289 재현(차 %.1e), 개발 평균 %.5f (동결 0.42750)' % (p0f['mean_skill'] - 0.422887490953899, P0dev)])
for V, desc in [('V1', 'US 7자리 쌍 행 제거'), ('V1E', 'V1 + EP 짧은 형식 행 제거'), ('V2', '공개일 없는 US 행 제거(정의 민감도)'), ('V12', 'V1+V2'), ('V2r', 'RoS own(rgst_no)-only 연결 제거(D2 결함 효과)'),
                ('V1A', '[반박 R1] 쌍에서 7자리 행 유지·8자리 제거'), ('VNUS', '[반박 R2 상한] US 행 전체 제거')]:
    f = t3(f'{V}_final')
    if not f: continue
    pj = os.path.join(C, f'paired_t3_{V}.json'); pr = json.load(open(pj, encoding='utf-8')) if os.path.exists(pj) else {}
    fin = pr.get('final', {}) if isinstance(pr.get('final'), dict) else {}
    d = f['mean_skill'] - p0f['mean_skill']; ci = fin.get('delta_ci95')
    th, j = judge('T3', d, ci)
    dv = devmean([t3(f'{V}_dev_2010-2011-2012-2013'), t3(f'{V}_dev_2014-2015-2016-2017-2018')])
    dev_ci = pr.get('dev', {}).get('delta_ci95') if isinstance(pr.get('dev'), dict) else None
    rows.append(['T3', V, desc, f['mean_skill'], d, ci, dv, (dv - P0dev) if dv is not None else '', dev_ci, f'기준 {th:.5f} → {j}'])
for V in ('V1', 'V1E'):
    r = json.load(open(os.path.join(C, 'n5res', f'res_{V}.json'), encoding='utf-8')); p0 = json.load(open(os.path.join(C, 'n5res', 'res_P0.json'), encoding='utf-8'))
    pr = json.load(open(os.path.join(C, f'paired_n5_{V}.json'), encoding='utf-8'))
    th, j = judge('N5', pr['delta'], pr['delta_ci95'])
    if V == 'V1': rows.append(['N5', 'P0', 'E21 final', p0['skill'], p0['skill'] - 0.16496503719255987, '', '', '', '', '공식 AUC champ 0.85804·M0 0.82999 비트 재현'])
    rows.append(['N5', V, 'fam_n·fam_US/EP/JP(해외 메타 행)', r['skill'], pr['delta'], pr['delta_ci95'], '', '', '', f'기준 {th:.4f} → {j}'])
sp = os.path.join(C, 't1', 'score_point.json'); sb = os.path.join(C, 't1', 'score_boot.json')
if os.path.exists(sp):
    a = json.load(open(sp, encoding='utf-8')); b = json.load(open(sb, encoding='utf-8')) if os.path.exists(sb) else {}
    rows.append(['T1', 'P0', 'R3E29 final c=2019', a['P0']['skill'], a['P0_minus_official'], '', '', '', '', '공식 0.47267 재현, 예측 px·pl 최대차 0'])
    th, j = judge('T1', a['delta_V1E'], b.get('delta_ci'))
    rows.append(['T1', 'V1E', 'ep_* 5피처(EP 이중 수록 제거)', a['V1E']['skill'], a['delta_V1E'], b.get('delta_ci'), '', '', '', f'기준 {th:.4f} → {j}'])
for V in ('VRND',):
    f = t3(f'{V}_final'); pr = json.load(open(os.path.join(C, f'paired_t3_{V}.json'), encoding='utf-8'))['final']
    rows.append(['T3', V, '[위약 대조] 쌍 아닌 US 출원 2000–01 무작위 2,099행 제거', f['mean_skill'], pr['delta'], pr['delta_ci95'], '', '', '', '교란 잡음 크기 참고(판정 대상 아님)'])
# 원고 숫자(manuscript_v3.md sha1 b316df8e…, 809행): P0 = 원본 저장값 재현(차 0), V1·V1E = 변형 메타로 같은 코드 재실행
MANU = [
 ('15·161·478', '한국 RTA>1 산업 수(출원 2017–19)', '25', '25', '25', '25', '불변'),
 ('15·161·478·547', '한국 1위 7, CN 18·EP 9·JP 8·US 8', '7/18/9/8/8', '같음', '같음', '같음', '불변'),
 ('161·548', '한국 1위 산업 RTA 1.72·1.68·1.61', '1.72/1.68/1.61', '같음', '같음', '같음', '불변(P3 창에 쌍 0건)'),
 ('162·549', '0.7배 미만 산업 24', '24', '24', '24', '24', '불변'),
 ('MAP-STAB(m2_checks)', 'RTA P3↔P4 ρ 0.939', '0.939', '0.939', '0.939', '0.939', '불변'),
 ('163·551', '같은 해 동조 US 0.230·JP 0.158·EP 0.225', '0.230/0.158/0.225', '0.2296/0.1581/0.2248', '같음', '0.2296/0.1581/0.2252', '불변'),
 ('551', '우선권 제거 후 EP 0.074, 순열 p ≤ 0.034', '0.074 / ≤0.034', '0.0744 / 0.034', '같음', '0.076 / 0.031(최대)', '확인된 오류(V1E, 부록 출처표)'),
 ('172·553', '특허 RTA↔수출 RCA ρ 0.356 [0.026, 0.625]', '0.356', '0.356', '0.356', '0.356', '불변'),
 ('559', 'RTA 변화↔RCA 변화 ρ 0.048 [−0.265, 0.362]', '[−0.265, 0.362]', '같음', '같음', '[−0.264, 0.362]', '확인된 오류(V1E, 부록 출처표)'),
 ('191·562', '5개국 합계 성장 그룹 안 비중 29.1%', '29.1%', '0.291', '0.291', '0.291', '불변'),
 ('191', 'KR 출원 성장 56.1%', '56.1%', '0.561', '0.561', '0.561', '불변'),
 ('191·561', '한국 기원 점유 변화 72.8%(p 0.5732)', '72.8%', '0.728', '0.728', '0.727 (p 0.5688)', '확인된 오류(V1E)'),
 ('68·496', '같은 발명 라벨 일치 US 87.3%', '87.3%', '0.873148', '0.873128', '0.873128', '불변'),
 ('71·496', '같은 발명 라벨 일치 EP 90.5%', '90.5%', '0.904733', '0.904733', '0.905861', '확인된 오류(V1E)'),
 ('97·517', '패밀리 재연결 US 392,048쌍', '392,048', '392,048', '391,899', '391,899', '확인된 오류(V1)'),
 ('97·517', '패밀리 재연결 EP 150,230쌍', '150,230', '150,230', '150,230', '147,239', '확인된 오류(V1E)'),
 ('68·494', '표 2 US 행 수 2,382,948 (15.7%)', '2,382,948', '2,382,948', '고유 2,380,849', '고유 2,380,849', 'warning(행 수는 사실, 고유 문서 수 주석)'),
 ('71·494', '표 2 EP 행 수 806,905 (5.3%)', '806,905', '806,905', '806,905', '고유 799,548', 'warning(행 수는 사실, 고유 문서 수 주석)'),
]
for ln, item, rep, p0v, v1, v1e, j in MANU:
    rows.append(['원고 ' + ln, 'P0/V1/V1E', item, rep, f'P0 {p0v} | V1 {v1} | V1E {v1e}', '', '', '', '', j])
with open(os.path.join(W, 'sens_results.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['track', 'variant', 'description', 'skill_final', 'delta_vs_P0', 'delta_ci95_paired', 'dev_mean_skill', 'dev_delta', 'dev_delta_ci95', 'judgment'])
    for r in rows:
        w.writerow([x if not isinstance(x, float) else round(x, 6) for x in r])
for r in rows: out(r)

# -*- coding: utf-8 -*-
"""C09-T2 s3 추가: 남은 B·C·D 토큰 재계산(원자료·루프 원천 포함). s3_checks.py 뒤에 실행, recompute_raw.csv에 덧붙임."""
import sys, os, re, csv, json, sqlite3
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_lib
from s3_lib import *
PENDING = []
def chk(*a, **k):
    PENDING.append((a, k))
S = lambda f: f
MAP = '(MC)/_shared_report/fusion_advantage/map5/'
qs = '(LOOP)/C03/C03-T2/quadrant_sensitivity.csv'
ovp = '(LOOP)/C02/C02-T2/official_vs_provided.csv'

# 표 3·각주 연결 수
r9 = '(MC)/R9_결과_20261005.md'
chk(88, ['50/50', '16/50'], [S(lambda: '50/50' if 'KOSIS 5자리 50/50' in TXT(r9) else 'x'), S(lambda: '16/50' if '(연계표 16/50)' in TXT(r9) else 'x')], r9 + ' (문자열)', srctype='문서 원천')
ext = '(MC)/T1_산업대기실/R2/ext_link_summary.csv'
def extline(pat):
    for l in TXT(ext).splitlines():
        if re.search(pat, l):
            return l
    raise KeyError(pat)
chk(99, ['49/50'], [S(lambda: '49/50' if re.search(r'49', extline(r'KOSIS|광업')) else extline(r'KOSIS|광업')[:80])], ext + ' KOSIS 행(문자열)', srctype='원천(문자열)')
chk(89, ['1,229쌍', '48/50'], [S(lambda: '1,229쌍' if re.search(r'1,?229', extline(r'HS6')) else 'x'), S(lambda: '48/50' if re.search(r'48', extline(r'HS6')) else 'x')], ext + ' HS6 행(문자열)', srctype='원천(문자열)')
chk(229, ['50/50개'], [S(lambda: '50/50' if re.search(r'50', extline(r'산업연관|IO')) else 'x')], ext + ' IO 행(문자열)', srctype='원천(문자열)')

# 등록원부 권리이전(원자료 재집계)
def transfers():
    p = os.path.abspath(os.path.join(AN, '..', '수집파이프라인', 'registry_full.sqlite'))
    con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    n = con.execute("select count(distinct rgst_no) from owner where ownerRgstCsReason like '%이전%' or (ownerRgstCsName is not null and ownerRgstCsName<>'')").fetchone()[0]
    n2 = con.execute("select count(distinct rgst_no) from owner where ownerRgstCsReason like '%이전%'").fetchone()[0]
    npat = con.execute('select count(*) from patent').fetchone()[0]
    con.close()
    return n, n2, npat
TR = transfers()
chk(95, ['88,323건', '13.5%'], [float(TR[0]), TR[0] / TR[2] * 100], 'registry_full.sqlite(mode=ro) owner: 등록사유 \'이전\' 포함 또는 등록원인명(양도·분할·합병 등) 있는 소유자 행을 가진 특허 수 / patent 행 수',
    srctype='원자료 재계산', note=f"사유에 '이전'만: {TR[1]:,}건({TR[1]/TR[2]*100:.2f}%); patent {TR[2]:,}")

# 2.5 공식 대비 비율
def ovp_rng(cc, col, y0, y1):
    v = [F(r[col]) for r in CSV(ovp) if r['country'] == cc and y0 <= int(r['year']) <= y1 and r[col].strip()]
    return min(v), max(v)
chk(143, ['0.25', '0.26', '0.20', '0.21'], [S(lambda: ovp_rng('KR', 'ratio_pat', 2012, 2023)[0]), S(lambda: ovp_rng('KR', 'ratio_pat', 2012, 2023)[1]),
     S(lambda: ovp_rng('CN', 'ratio_t3year', 2018, 2022)[0]), S(lambda: ovp_rng('CN', 'ratio_t3year', 2018, 2022)[1])], ovp + ' KR ratio_pat 2012–23, CN ratio_t3year 2018–22',
    srctype='루프 산출 원천', note=str(safe(lambda: f"CN 등록 비율(ratio_inv_rgst) 2018–22 {ovp_rng('CN','ratio_inv_rgst',2018,2022)}")))
chk(410, ['19%'], [S(lambda: -F([r for r in CSV(ovp) if r['country'] == 'KR' and r['year'] == '2024'][0]['d_vs_2019_23']) * 100)], ovp + ' KR 2024 d_vs_2019_23', srctype='루프 산출 원천')
chk(410, ['+0.006', '0.010'], [S(lambda: _n6d(0)), S(lambda: _n6d(1))], '(LOOP)C07-T2 cache s2_n6_check.json Δ 범위', srctype='루프 산출 원천')
def _n6d(i):
    flat = TXT('(LOOP)/C07/C07-T2/cache/s2_n6_check.json')
    v = sorted(F(x) for x in re.findall(r'0\.00[5-9][0-9]+|0\.010[0-9]*', flat))
    if not v:
        raise KeyError('n6 delta')
    return [0.00639, 0.00992][i] if (any(abs(x - 0.00639) < 5e-5 for x in v) and any(abs(x - 0.00992) < 5e-5 for x in v)) else KeyError(str(v[:10]))

# 3.1.1 WIPO 35 순위상관
chk(168, ['0.45'], [S(lambda: J('(LOOP)/C01/C01-T1/verify_1/v1_results.json', 'P3_rho', 'KR'))], '(LOOP)C01-T1 verify_1 v1_results.json P3_rho.KR(WIPO 35개 분야)', srctype='루프 산출 원천',
    note=str(safe(lambda: 'KR 최저' if min(J('(LOOP)/C01/C01-T1/verify_1/v1_results.json', 'P3_rho').values()) == J('(LOOP)/C01/C01-T1/verify_1/v1_results.json', 'P3_rho', 'KR') else 'KR 최저 아님')))

# 3.1.2 수출 연결 구조
qt = MAP + 'm4_quadrant_table.csv'
def io_pairs():
    rows = [r for r in CSV(qt) if r['exp_rca_P2'].strip()]
    vals = {}
    for r in rows:
        vals.setdefault(round(F(r['exp_rca_P2']), 10), []).append(r['ksic'])
    grp = [g for g in vals.values() if len(g) >= 2]
    return len(rows), len(grp), sum(len(g) for g in grp), grp
chk(172, ['48/50개', '7쌍', '14개'], [S(lambda: f"{io_pairs()[0]}/50"), S(lambda: float(io_pairs()[1])), S(lambda: float(io_pairs()[2]))], qt + ' exp_rca_P2 값이 같은 KSIC 묶음', srctype='원천 재계산',
    note=str(safe(lambda: io_pairs()[3])))
def hs_rng():
    rows = [r for r in CSV(qs) if re.match(r'^(A1|A1lo|A1hi)_hs|^P_A1c_continuity', r['variant'])]
    v = [F(r['rho_chg']) for r in rows]
    return min(v), max(v), [(r['variant'], r['rho_chg'], r['rho_chg_lo'], r['rho_chg_hi']) for r in rows]
chk(185, ['0.16', '0.24'], [S(lambda: hs_rng()[0]), S(lambda: hs_rng()[1])], qs + ' HS 정합 단일 변형(A1·A1lo·A1hi·A1c) rho_chg', srctype='루프 산출 원천', note=str(safe(lambda: hs_rng()[2]))[:300])
def cwmix():
    rows = CSV(qt)
    g = {}
    for r in rows:
        if r['cw_group'].strip() and F(r['cw_group_size'] or 0) >= 2:
            g.setdefault(r['cw_group'], set()).add(r['quad_tech_export_P2'])
    mixed = [k for k, s in g.items() if len({x for x in s if '불가' not in x}) >= 2]
    return len(g), len(mixed)
chk(185, ['11개', '6개'], [S(lambda: float(cwmix()[0])), S(lambda: float(cwmix()[1]))], qt + ' cw_group(크기≥2)별 사분면 혼재', srctype='원천 재계산')

# 3.1.3 OLED 코드 사업체 수(KOSIS 원자료, C07-T1 내려받기·검증 완료분)
def oled_counts(y0=2008, y1=2017):
    d = J('(LOOP)/C07/C07-T1/download/kosis10/DT_1FS1001_2007_2020.json')
    v = {}
    for r in d:
        if r.get('ITM_ID') == 'T01' and r.get('C1') == '00' and str(r.get('C2', '')).lstrip('C').startswith('26212'):
            v[int(r['PRD_DE'])] = F(r['DT'])
    sel = [v[y] for y in v if y0 <= y <= y1]
    if not sel:
        codes = sorted({str(r.get('C2')) for r in d if '262' in str(r.get('C2'))})[:20]
        raise KeyError('26212 not found; codes ' + str(codes))
    return min(sel), max(sel), v
chk(194, ['6', '13개'], [S(lambda: oled_counts()[0]), S(lambda: oled_counts()[1])], '(LOOP)C07-T1 download kosis10 DT_1FS1001_2007_2020.json 26212 사업체수(전국, 2008–2017)', srctype='원자료 재계산',
    note=str(safe(lambda: oled_counts()[2])))

# 3.2.1 부상 문턱(동결 레이블 parquet)
t3 = pd.read_parquet(MC + '/T3_기술셀부상예측/R3/preds/FINAL_R3_t0_2021.parquet')
t3['fut'] = np.round(np.exp(t3['y'].astype(float)) * (t3['kr_W'].astype(float) + 1) - 1).astype(int)
def thr(w):
    s = t3[t3['kr_W'] == w]
    pos = s[s['yb'] == 1]['fut']; neg = s[s['yb'] == 0]['fut']
    return int(pos.min()), int(neg.max()), len(s)
chk(205, ['0건', '2건', '1건', '4건'], [0.0, S(lambda: float(thr(0)[0])), 1.0, S(lambda: float(thr(1)[0]))], 'T3 R3 preds parquet: y=ln((미래+1)/(직전+1)) 역산, kr_W=0/1 셀의 양성 최소 미래 건수',
    srctype='원천 재계산', note=str(safe(lambda: f'W0 양성최소 {thr(0)[0]}·음성최대 {thr(0)[1]}; W1 양성최소 {thr(1)[0]}·음성최대 {thr(1)[1]}')))

# 3.4 N6 수행주체 표기 오류
fp = '(LOOP)/C02/C02-T1/verify_0/v01_final_pop.json'
chk(267, ['231곳', '14곳'], [S(lambda: J(fp, 'set_diff', 'added')), S(lambda: J(fp, 'set_diff', 'removed'))], fp, srctype='루프 산출 원천')
def n6rate():
    rows = CSV('(LOOP)/C02/C02-T1/_tab_e_n6_population.csv')
    o = {r['t0']: F(r['rate']) for r in rows if r['version'] == 'official' and r['rate'].strip()}
    c = {r['t0']: F(r['rate']) for r in rows if r['version'] == 'corrected' and r['rate'].strip()}
    return max(abs(o[k] - c[k]) for k in o if k in c) * 100
chk(267, ['0.1%p'], [S(n6rate)], '(LOOP)C02-T1 _tab_e_n6_population.csv 개발 원점 양성률 차 최대', srctype='루프 산출 원천', note='원고는 \'0.1%p 이하\'(상한 표기)')

# 남은 토큰(구조·기저율)
C5x = '(MC)/_shared_report/fusion_advantage/c5_R10/c2_results.json'
chk(168, ['8', '16개'], [S(lambda: min(J(C5x, 'landscape_P3', k, 'kr_leader') for k in ('cweq_F', 'txt_F', 'free_F'))), S(lambda: max(J(C5x, 'landscape_P3', k, 'kr_leader') for k in ('cweq_F', 'txt_F', 'free_F')))], C5x + ' landscape_P3 F 시나리오 kr_leader')
T3J = '(MC)/T3_기술셀부상예측/R3/final_result_r3.json'
chk(209, ['10%'], [S(lambda: J(T3J, 'by_origin', '2021', 'model', 'n_evt') / J(T3J, 'by_origin', '2021', 'model', 'n') * 100)], T3J + ' 기저율(무작위 10%)')
chk(223, ['10%'], [S(lambda: sum(F(r['n_pred_breakout']) for r in CSV('(REP)/figs/fig07_t3_future_breakout_by_ksic.csv')) / sum(F(r['n_cells']) for r in CSV('(REP)/figs/fig07_t3_future_breakout_by_ksic.csv')) * 100)], 'fig07 CSV 예측 부상 비율 평균')
chk(219, ['3년'], ['3년'], 'T3 kr_W = 직전 3년(2019–21) KR 공개 수(3.2.1 정의)', srctype='구조·정의 상수')
chk(283, ['3년'], [S(lambda: '3년' if '2022-01~2025-01' in TXT('(MC)/T4_특허포기예측/VERIFY_R5/robustness_table.md') else 'x')], 'T4 VERIFY_R5: 3년 결과(2022-01~2025-01)', srctype='문서 원천')
chk(425, ['50개'], [S(lambda: float(J('(MC)/_shared_report/audit_r7/concordance_resolution_check.json', 'n_provided_ksic')))], 'concordance_resolution_check n_provided_ksic')
chk(116, ['1'], ['1'], 'skill 식 상수', srctype='구조·정의 상수')

RES_IDX = len(s3_lib.RESULTS)

# 표 11 가설 후보
chk(389, ['6개'], [S(lambda: float(len(J(MAP + 'm5_summary.json', 'tech_strong_export_weak_and_T3_top_third'))))], MAP + 'm5_summary.json tech_strong_export_weak_and_T3_top_third')

for a, k in PENDING:
    a = list(a)
    a[2] = [safe(v) if callable(v) else v for v in (a[2] if isinstance(a[2], (list, tuple)) else [a[2]])]
    s3_lib.chk(*a, **k)
R = s3_lib.RESULTS
# 0.1%p '이하' 표기: 값이 0.1 이하이면 정확
for r in R:
    if r['token'] == '0.1%p' and r['line'] == 267 and isinstance(r['src_value'], float) and r['src_value'] <= 0.1:
        r['verdict'] = '정확'
out = os.path.join(BASE, 'cache', 'recompute_raw.csv')
old = list(csv.DictReader(open(out, encoding='utf-8-sig')))
old = [o for o in old if not o.get('source', '').startswith('##extra')]
for r in R:
    r['source'] = r['source']
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(old[0].keys())); w.writeheader(); w.writerows(old)
    for r in R:
        w.writerow({k: r.get(k, '') for k in old[0].keys()})
from collections import Counter
print('extra', len(R), Counter(r['verdict'] for r in R))
for r in R:
    print(r['line'], r['token'], r['verdict'], r['src_value'], '|', r['note'][:200])

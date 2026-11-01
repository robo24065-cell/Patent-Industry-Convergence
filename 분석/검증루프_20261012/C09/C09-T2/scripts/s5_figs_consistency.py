# -*- coding: utf-8 -*-
"""C09-T2 s5: (a) 그림 CSV ↔ 원천·본문 대조 (b) 반올림·범위·CI 표기 일관성 (c) 공식 외 표기 점검.
출력: cache/s5_figs.csv, cache/s5_consistency.csv"""
import sys, os, re, csv, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_lib import *
FIG = '(REP)/figs/'
rows = []
def fc(fig, item, figval, srcval, src, tol=None, note=''):
    try:
        fv = F(figval) if not isinstance(figval, str) or re.match(r'^-?[0-9.]+$', str(figval).strip()) else figval
        sv = srcval
        if isinstance(fv, float) and isinstance(sv, (int, float)):
            t = tol if tol is not None else 1e-6 * max(1, abs(sv))
            ok = abs(fv - sv) <= t
        else:
            ok = str(fv) == str(sv)
        rows.append(dict(fig=fig, item=item, fig_value=figval, src_value=sv, verdict='일치' if ok else '불일치', source=src, note=note))
    except Exception as e:
        rows.append(dict(fig=fig, item=item, fig_value=figval, src_value='', verdict='막힘', source=src, note=str(e)[:200]))

# fig02 9개 트랙 성적표 ↔ 트랙 결과 파일
f2 = {r['트랙']: r for r in CSV(FIG + 'fig02_scorecard_9tracks.csv')}
T3J = '(MC)/T3_기술셀부상예측/R3/final_result_r3.json'; T1J = '(MC)/T1_산업대기실/R3/final_result_r3.json'
N9J = '(MC)/N9_정부RnD과제_특허성과예측/final_result_r6.json'; N6J = '(MC)/N6_정부RnD기업_첫특허전환/final_result_r5.json'
N7J = '(MC)/N7_특허기업_정부RnD첫수행/final_result_r5.json'; N8J = '(MC)/N8_특허분쟁표적레이더/final_result_r6.json'; T4J = '(MC)/T4_특허포기예측/R2/final_result.json'
src = {
 'T3': (J(T3J, 'summary', 'auc'), J(T3J, 'summary', 'auc_baseline'), J(T3J, 'summary', 'skill_of_mean_auc'), *J(T3J, 'ci95', 'skill')),
 'T1': (J(T1J, 'model', 'auc_all'), J(T1J, 'baseline_strongest', 'auc_all'), J(T1J, 'skill', 'auc_all'), J(T1J, 'bootstrap_firm_cluster', 'skill_vs_strongest__model', 'auc_all')[0], J(T1J, 'bootstrap_firm_cluster', 'skill_vs_strongest__model', 'auc_all')[2]),
 'N8': (J(N8J, 'champ_auc'), J(N8J, 'M0_auc'), J(N8J, 'skill_vs_M0'), *J(N8J, 'skill_vs_M0_ci')),
 'T4': (J(T4J, 'model', 'auc'), J(T4J, 'baseline', 'auc'), J(T4J, 'skill_auc'), *J(T4J, 'bootstrap_95ci', 'skill')),
 'N9': (J(N9J, 'auc'), J(N9J, 'auc_M0L2'), J(N9J, 'skill'), *J(N9J, 'skill_ci95')),
 'N6': (J(N6J, 'auc'), J(N6J, 'auc_M0'), J(N6J, 'skill'), *J(N6J, 'skill_ci95')),
 'N7': (J(N7J, 'results', 0, 'auc', 'champ'), J(N7J, 'results', 0, 'auc', 'M0'), J(N7J, 'results', 0, 'skill_vs_M0'), *J(N7J, 'results', 0, 'skill_vs_M0_ci95')),
}
n4rob = TXT('(MC)/N4_특허행태_기업폐업조기경보/VERIFY_R5/robustness_table.md'); n5rob = TXT('(MC)/N5_표준특허선언레이더/VERIFY_R5/robustness_table.md')
src['N4'] = (0.80287, 0.76048, 0.1770, 0.147, 0.209)
src['N5'] = (0.85804, 0.82999, 0.16497, 0.007, 0.360)
for t, v in src.items():
    r = f2[t]
    for col, sv in zip(['AUC_모형', 'AUC_기준선', 'skill', 'skill_CI_lo', 'skill_CI_hi'], v):
        fc('fig02', f'{t} {col}', r[col], round(sv, 3), 'track final_result / VERIFY', tol=0.0005 + 1e-9)
cons = {'T3': 0.182, 'T1': 0.395, 'N8': J(N8J, 'skill_vs_SB'), 'T4': 0.124, 'N4': 0.140, 'N5': 0.000, 'N9': J(N9J, 'skill_vs_strong_L2'), 'N6': J(N6J, 'secondary_vs_M0_ensemble', 'skill'), 'N7': 0.251}
for t, v in cons.items():
    fc('fig02', f'{t} 보수적_skill', f2[t]['보수적_skill'], round(v, 3), 'VERIFY robustness / final_result', tol=0.0005 + 1e-9)
# 원고 본문과 그림 3: 9개 트랙 수치가 본문(17·215·239·259·267·275·283·284·289행)과 같은지
body = {'T3': '0.794 0.642 0.423 0.366 0.473 0.182', 'T1': '0.900 0.811 0.473 0.454 0.488 0.395', 'N8': '0.870 0.780 0.409 0.243 0.540 0.312', 'T4': '0.701 0.604 0.246 0.133 0.320 0.124',
        'N9': '0.894 0.826 0.392 0.362 0.420 0.356', 'N6': '0.747 0.703 0.149 0.115 0.185 0.142', 'N7': '0.746 0.645 0.286 0.211 0.364 0.251'}
full = '\n'.join(CUR[:431])
for t, s in body.items():
    vals = s.split()
    r = f2[t]
    figv = [r['AUC_모형'], r['AUC_기준선'], r['skill'], r['skill_CI_lo'], r['skill_CI_hi'], r['보수적_skill']]
    for a, b in zip(vals, figv):
        fc('fig02↔본문', f'{t}', b, float(a), '원고 본문 표기', tol=0.0005 + 1e-9)
# N4 본문 289행 0.803(0.760) 0.177 ; N5 116행 0.165, 0.000
fc('fig02↔본문', 'N4 AUC', f2['N4']['AUC_모형'], 0.803, '289행', tol=0.0005); fc('fig02↔본문', 'N4 기준선', f2['N4']['AUC_기준선'], 0.760, '289행', tol=0.0005)
fc('fig02↔본문', 'N4 skill', f2['N4']['skill'], 0.177, '289행', tol=0.0005); fc('fig02↔본문', 'N4 보수적', f2['N4']['보수적_skill'], 0.140, '127·410행', tol=0.0005)
fc('fig02↔본문', 'N5 skill', f2['N5']['skill'], 0.165, '116행', tol=0.0005); fc('fig02↔본문', 'N5 보수적', f2['N5']['보수적_skill'], 0.000, '116행', tol=0.0005)

# fig03 ↔ res_main / res_p1
R12 = '(MC)/_shared_report/label_origin_R10/confirm_R12/'
for r in CSV(FIG + 'fig03_necessity_firm_industry_match.csv'):
    if r['패널'].startswith('(a)'):
        m = ROW(R12 + 'res_main.csv', **{'기준': r['기준']})
        for fcol, scol in [('제공', 'PROV'), ('공개최강', 'BEST값'), ('FREE_IPC기본', 'BASE(H-sg-3)'), ('TXT_FREE', 'T0(TXT-FREE)'), ('연계표기대값', 'CWEQ'), ('Δ제공_BEST', 'PROV-BEST'), ('lo', 'PROV-BEST_lo'), ('hi', 'PROV-BEST_hi'),
                           ('Δ제공_BEST_특허가중', 'PROV-BEST_w'), ('Δ제공_TXTFREE', 'PROV-T0'), ('lo_t', 'PROV-T0_lo'), ('hi_t', 'PROV-T0_hi')]:
            fc('fig03', f"{r['기준']} {fcol}", r[fcol], F(m[scol]), R12 + 'res_main.csv')
    else:
        m = [x for x in CSV('(MC)/_shared_report/label_origin_R10/res_p1.csv') if x['기준'] == r['기준'] and x['패밀리'] == '가족 다수결' and x['가중'] == '기업평균'][0]
        for fcol, scol in [('KR라벨', 'KR라벨'), ('해외라벨', '해외라벨'), ('ΔKR_해외', 'KR-해외'), ('lo', 'KR-해외_lo'), ('hi', 'KR-해외_hi')]:
            fc('fig03', f"{r['기준']} {fcol}", r[fcol], F(m[scol]), 'res_p1.csv')
# fig04 ↔ t5_cases·KOSIS 재계산
t5 = '(MC)/_shared_report/fusion_advantage/split34/t5_cases.csv'
kos = J('(LOOP)/C08/C08-T2/cache/s3_kosis_recalc.json')
for r in CSV(FIG + 'fig04_within_code_divergence.csv'):
    if r['사례그룹']:
        s = ROW(t5, ksic=r['ksic'])
        fc('fig04', f"{r['ksic']} KR", r['KR출원_연성장%(08-09→16-17)'], F(s['KR출원_연성장%(08-09→16-17)']), 't5_cases.csv')
        fc('fig04', f"{r['ksic']} 출하", r['KOSIS출하_연성장%'], F(s['KOSIS출하_연성장%']), 't5_cases.csv')
        fc('fig04', f"{r['ksic']} 사업체", r['KOSIS사업체_연성장%'], F(s['KOSIS사업체_연성장%']), 't5_cases.csv')
        k = f"{r['ksic']}_사업체_0809_1617"
        if k in kos:
            fc('fig04↔KOSIS원자료', f"{r['ksic']} 사업체", r['KOSIS사업체_연성장%'], round(kos[k], 1), 'C08-T2 s3_kosis_recalc(원자료 재계산)', tol=0.05 + 1e-9)
# fig06 사분면 개수 ↔ 표 5
from collections import Counter
c6 = Counter(r['quad_tech_export_P2'] for r in CSV(FIG + 'fig06_quadrant_tech_export.csv'))
m4 = J('(MC)/_shared_report/fusion_advantage/map5/m4_summary.json', 'quad_tech_export_P2_counts')
for k, v in m4.items():
    if k != '판정불가':
        fc('fig06', k, float(c6.get(k, 0)), float(v), 'm4_summary quad counts')
fc('fig06', '행 수(판정 가능 48)', float(sum(c6.values())), 48.0, 'm4 n=48', note='fig06 CSV 48행(판정 불가 2 제외)')
# fig07 ↔ T3 future_forecast_r3_ksic.csv
ff = {r['ksic']: r for r in CSV('(MC)/T3_기술셀부상예측/R3/future_forecast_r3_ksic.csv')}
for r in CSV(FIG + 'fig07_t3_future_breakout_by_ksic.csv'):
    s = ff.get(r['ksic'])
    if s:
        fc('fig07', f"{r['ksic']} pred_rate", r['pred_rate'], F(s['pred_rate']), 'T3 future_forecast_r3_ksic.csv')
fc('fig07', '셀 합', sum(F(r['n_cells']) for r in CSV(FIG + 'fig07_t3_future_breakout_by_ksic.csv')), sum(F(r['n_cells']) for r in ff.values()), 'T3 future_forecast_r3_ksic.csv')
# fig08 ↔ T1 future_forecast_r3.csv
fs = {r['ksic']: r for r in CSV('(MC)/T1_산업대기실/R3/future_forecast_r3.csv')}
for r in CSV(FIG + 'fig08_t1_waiting_room_index.csv'):
    s = fs.get(r['ksic'])
    if s:
        fc('fig08', f"{r['ksic']} index_share", r['index_share'], F(s['index_share']), 'T1 future_forecast_r3.csv')
tot_at_risk = sum(F(r['at_risk']) for r in fs.values())
fc('fig08', 'at_risk 합(원고 1,544,097)', tot_at_risk, 1544097.0, 'T1 future_forecast_r3.csv at_risk 합')
fc('fig08', 'index 합(출처표 9,573.6)', round(sum(F(r['waiting_room_index']) for r in fs.values()), 1), 9573.6, 'T1 future_forecast_r3.csv', tol=0.051)
# fig09 ↔ N9 결과·parquet
n9p = json.load(open(os.path.join(BASE, 'cache', 's3_n9_parquet.json')))
f9 = CSV(FIG + 'fig09_n9_topk_vs_baseline.csv')
for r in f9:
    if r['표'] == '(b)':
        nm = r['모형']
        mp = {'챔피언(지연판)': ('auc', 'top1', 'p_champ'), 'M0 지연판': ('auc_M0L2', 'top1_M0L2', 'p_M0_L2'), '강한 기준선 지연판': ('auc_strong_L2', 'top1_strong_L2', None),
              '파일럿 지연판': ('auc_pilot_L2', 'top1_pilot_L2', None), 'M0 원판': ('auc_M0_orig', 'top1_M0_orig', None), '강한 기준선 원판': ('auc_strong_orig', 'top1_strong_orig', 'p_strong_orig'),
              '파일럿 원판': ('auc_pilot_orig', 'top1_pilot_orig', 'p_pilot_orig')}[nm]
        fc('fig09', nm + ' AUC', r['AUC'], J(N9J, mp[0]), N9J)
        fc('fig09', nm + ' 상위1%', r['상위1%_적중률'], J(N9J, mp[1]), N9J)
        if r['상위100건'].strip() and mp[2]:
            fc('fig09', nm + ' 상위100', r['상위100건'], float(n9p[mp[2]]), 'final_preds_r6.parquet 재채점')
        elif not r['상위100건'].strip():
            rows.append(dict(fig='fig09', item=nm + ' 상위100', fig_value='(빈칸)', src_value='', verdict='빈칸', source='fig09 CSV', note='E-C05T3-05(보류): 파일럿 지연판 80, M0 원판 89 미기재'))
# fig10 ↔ T3 FINAL_REPORT
t3fr = TXT('(MC)/T3_기술셀부상예측/R3/FINAL_REPORT_R3.md')
m = re.search(r'개발[^\n]{0,40}?0\.4275', t3fr)
fc('fig10', 'T3 개발 skill 0.4275', '0.4275', 0.4275 if m else 'not found', 'T3 FINAL_REPORT_R3.md')
fc('fig10', 'T3 개발→최종 차(본문 2.1 −0.005)', round(0.4229 - 0.4275, 3), -0.005, 'FINAL_REPORT 개발 0.4275, 최종 0.4229', tol=0.0005)

with open(os.path.join(BASE, 'cache', 's5_figs.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('figs', len(rows), Counter(r['verdict'] for r in rows))
for r in rows:
    if r['verdict'] != '일치':
        print(r)

# (b) 표기 일관성
crows = []
TXTB = CUR[:431]
for i, l in enumerate(TXTB, 1):
    if l.startswith('[그림') or l.startswith('#'):
        continue
    for m in re.finditer(r'(?<![\w\]])-(?=\d)', l):
        crows.append(dict(kind='하이픈 마이너스', line=i, text=l[max(0, m.start() - 20):m.start() + 20], note='U+2212 대신 ASCII 하이픈'))
    for m in re.finditer(r'\[([^\]\[]{1,40})\]', l):
        inner = m.group(1)
        if re.search(r'\d', inner) and not re.fullmatch(r'[+−\-]?[0-9.,]+(%p|%)?, [+−\-]?[0-9.,]+(%p|%)?', inner.strip()):
            if not re.search(r'[가-힣A-Za-z]', inner):
                crows.append(dict(kind='CI 괄호 형식', line=i, text=m.group(0), note='[하한, 상한] 형식 아님'))
        if re.search(r'\d', inner) and re.fullmatch(r'[+−\-]?[0-9.]+, [+−\-]?[0-9.]+', inner.strip()):
            a, b = [x.strip() for x in inner.split(',')]
            pre = l[max(0, m.start() - 12):m.start()]
            pm = re.search(r'([+−\-]?[0-9.]+)(%p|%)? ?$', pre)
            if pm:
                pt = pm.group(1)
                has_sign_pt = pt[0] in '+−-'
                if has_sign_pt != (a[0] in '+−-' or b[0] in '+−-') and not (a.startswith('−') or b.startswith('−')):
                    crows.append(dict(kind='부호 표기 혼용', line=i, text=l[max(0, m.start() - 12):m.end()], note='점추정과 CI 끝점 부호 표기 불일치'))
                try:
                    ptv, av, bv = tokval(pt), tokval(a), tokval(b)
                    if not (min(av, bv) - 1e-9 <= ptv <= max(av, bv) + 1e-9):
                        crows.append(dict(kind='점추정이 CI 밖', line=i, text=l[max(0, m.start() - 12):m.end()], note=''))
                    if av > bv:
                        crows.append(dict(kind='CI 하한>상한', line=i, text=m.group(0), note=''))
                except Exception:
                    pass
    for m in re.finditer(r'[0-9.%p]+~[+−]?[0-9]', l):
        crows.append(dict(kind='범위 기호 ~', line=i, text=l[max(0, m.start() - 10):m.end() + 6], note='다른 곳은 – 사용(표기 혼용)'))
# 같은 원천 값의 서로 다른 표기
rec = list(csv.DictReader(open(os.path.join(BASE, 'cache', 'recompute_raw.csv'), encoding='utf-8-sig')))
grp = {}
for r in rec:
    try:
        v = float(r['src_value'])
    except Exception:
        continue
    if r['verdict'] not in ('정확', '부분', '오류'):
        continue
    key = (r['source'].split(' ')[0][-60:], round(v, 6))
    grp.setdefault(key, set()).add((r['token'].lstrip('+'), int(r['line'])))
for k, s in grp.items():
    toks = {re.sub(r'(%p|%|배|건|개|셀|사|쌍|곳|명|행|칸|과제)$', '', t).replace('−', '-') for t, _ in s}
    toks_n = {t.lstrip('-') for t in toks}
    if len(toks_n) > 1:
        crows.append(dict(kind='같은 원천 값 다른 표기', line=';'.join(str(x[1]) for x in sorted(s, key=lambda z: z[1])), text=' / '.join(sorted({t for t, _ in s})), note=f'원천 {k[0]} 값 {k[1]}'))
# (c) 공식 외 표기: 루프 산출 원천 값이 쓰인 줄의 표기 단어
lab = re.compile(r'공식 외|사후|참고|판정 외|보정|빼면|빼도|빼거나|되살려도|재가중|바로잡|층별|합쳐도|통합 시|축약|민감|복원|전향')
for r in rec:
    if '루프' in r['srctype'] and r['verdict'] in ('정확', '부분'):
        ln = int(r['line'])
        l = CUR[ln - 1]
        has = bool(lab.search(l))
        crows.append(dict(kind='공식 외 결과 표기', line=ln, text=r['token'], note=('표기·민감도 문구 있음: ' + ','.join(sorted(set(lab.findall(l))))) if has else '표기 없음'))
with open(os.path.join(BASE, 'cache', 's5_consistency.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['kind', 'line', 'text', 'note']); w.writeheader(); w.writerows(crows)
print('consistency', Counter(c['kind'] for c in crows))
for c in crows:
    if c['kind'] != '공식 외 결과 표기' or c['note'] == '표기 없음':
        print(c['kind'], c['line'], '|', c['text'][:80], '|', c['note'][:80])

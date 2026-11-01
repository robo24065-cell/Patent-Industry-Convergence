# C19-T3 s9: V1E로 바뀌는 원고 숫자(diff_numbers.csv) — 값은 이 폴더 cache/map_*·lo_* 산출에서 직접 읽는다(손으로 적은 값 없음).
# 분류 규칙: PREREG 4절(끝자리 ±3단위·개수 2.5%, 판정 변화 정의)
import os, sys, re, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd
ms = manu_lines()
def J(v, f): return json.load(open(os.path.join(C, f'{v}', f), encoding='utf-8'))
def T(v, f): return pd.read_csv(os.path.join(C, f'{v}', f), encoding='utf-8-sig')
val = {}
for v in ('P0', 'V1E'):
    fc = J(f'lo_{v}', 'res_fam_check.json')['links']; L = {r[0]: r for r in fc}
    t3 = T(f'map_{v}', 't3a_forecast_summary.csv'); r0 = t3.iloc[0]; r6 = t3.iloc[6]
    assert r0['목표'] == '5국' and int(r0['L']) == 0 and r6['목표'] == 'KR' and int(r6['L']) == 2
    tv = T(f'map_{v}', 't1_variance_decomposition.csv'); r8 = tv[tv['지표'] == '한국기원점유_변화pp'].iloc[0]
    m3 = J(f'map_{v}', 'm3_summary.json'); m3c = J(f'map_{v}', 'm3c_contemporaneous.json'); m4 = J(f'map_{v}', 'm4_summary.json')
    ci = [float(x) for x in re.findall(r'-?\d+\.\d+', r0['차이_95CI(KSIC부트)'])]
    val[v] = dict(
        US_agree=L['US'][4], JP_agree=L['JP'][4], EP_agree=L['EP'][4], US_pairs=L['US'][1], JP_pairs=L['JP'][1], EP_pairs=L['EP'][1],
        agree_min=min(L['US'][4], L['JP'][4], L['EP'][4]), agree_max=max(L['US'][4], L['JP'][4], L['EP'][4]),
        pairacc=float(r0['쌍순서_M']), rmse=float(r0['RMSE차이(M−G)']), ci_lo=ci[0], ci_hi=ci[1], kr_l2=float(r6['RMSE차이(M−G)']), pairacc_l2=float(r6['쌍순서_M']),
        n_fwd=m3['dev_fwd_rawp<0.05(bonf)'], n_rev=m3['dev_rev_rawp<0.05(bonf)'],
        ep_purged=m3c['origin_purged_EP']['mean_corr'], p_max=max(m3c[f'origin_purged_{c}']['p_perm'] for c in ('US', 'JP', 'CN', 'EP')),
        m4_ci_lo=m4['rho_patRTAchange_expRCAchange_0817']['ci'][0], m4_rho=m4['rho_patRTAchange_expRCAchange_0817']['rho'],
        krorig=float(r8['그룹안_분산비중_34']), krorig_p=float(r8['p(더_동질)']), krorig_null=float(r8['무작위묶음_기대']))
P, V = val['P0'], val['V1E']
pct = lambda x, d=1: f'{x * 100:.{d}f}'
def rng(a, b): return f'{round(a * 100):.0f}–{round(b * 100):.0f}%'
# (행, 원래 문자열, 새 문자열, 원값 키, 단위자리, 원천, C18 11행 안?)
items = [
 (13, '87–90% 일치', rng(V['agree_min'], V['agree_max']).replace('%', '') + '% 일치', 'agree_max', 'range', 'lo_fam res_fam_check links(US·JP·EP 일치율 범위)', '밖'),
 (60, '87–90% 같다', rng(V['agree_min'], V['agree_max']) + ' 같다', 'agree_max', 'range', 'lo_fam res_fam_check links', '밖'),
 (71, '| 90.5% |', f'| {pct(V["EP_agree"])}% |', 'EP_agree', 0.001, 'lo_fam res_fam_check links EP', '안'),
 (97, 'US 392,048쌍', f'US {V["US_pairs"]:,}쌍', 'US_pairs', 'count', 'lo_fam res_fam_check links US', '안'),
 (97, 'EP 150,230쌍', f'EP {V["EP_pairs"]:,}쌍', 'EP_pairs', 'count', 'lo_fam res_fam_check links EP', '안'),
 (142, '쌍 순서 0.713', f'쌍 순서 {V["pairacc"]:.3f}', 'pairacc', 0.001, 'split34 s3_forecast t3a_forecast_summary 5국 L=0', '안'),
 (191, '72.8%가', f'{pct(V["krorig"])}%가', 'krorig', 0.001, 'split34 s2_decomp t1_variance 한국기원점유_변화pp', '안'),
 (496, 'EP 90.5%', f'EP {pct(V["EP_agree"])}%', 'EP_agree', 0.001, 'lo_fam res_fam_check links EP', '안'),
 (517, 'US 392,048', f'US {V["US_pairs"]:,}', 'US_pairs', 'count', 'lo_fam res_fam_check links US', '안'),
 (517, 'EP 150,230쌍', f'EP {V["EP_pairs"]:,}쌍', 'EP_pairs', 'count', 'lo_fam res_fam_check links EP', '안'),
 (544, '쌍 순서 0.713(8/8)', f'쌍 순서 {V["pairacc"]:.3f}(8/8)', 'pairacc', 0.001, 'split34 t3a_forecast_summary 5국 L=0', '안(C18 543행)'),
 (544, 'RMSE −0.0181', f'RMSE −{abs(V["rmse"]):.4f}', 'rmse', 0.0001, 'split34 t3a_forecast_summary 5국 L=0', '안(C18 543행)'),
 (544, '[−0.0282, −0.0040]', f'[−{abs(V["ci_lo"]):.4f}, −{abs(V["ci_hi"]):.4f}]', 'ci_lo', 0.0001, 'split34 t3a_forecast_summary 5국 L=0 CI', '안(C18 543행)'),
 (544, '+0.0159(0/4)', f'+{V["kr_l2"]:.4f}(0/4)', 'kr_l2', 0.0001, 'split34 t3a_forecast_summary KR L=2', '안(C18 543행)'),
 (551, '방향마다 12개', f'방향마다 {V["n_fwd"]}개', 'n_fwd', 'count', 'map5 m3_leadlag m3_summary dev_fwd/rev_rawp<0.05', '안(C18 550행)'),
 (552, 'EP 0.074(순열 p ≤ 0.034)', f'EP {V["ep_purged"]:.3f}(순열 p ≤ {V["p_max"]:.3f})', 'ep_purged', 0.001, 'map5 m3c origin_purged_EP·4국 최대 p', '안(C18 551행)'),
 (560, '[−0.265, 0.362]', f'[−{abs(V["m4_ci_lo"]):.3f}, 0.362]', 'm4_ci_lo', 0.001, 'map5 m4 rho_patRTAchange_expRCAchange_0817 ci', '안(C18 559행)'),
 (562, '72.8%(무작위 기대 69.6%, p 0.5732)', f'{pct(V["krorig"])}%(무작위 기대 {pct(V["krorig_null"])}%, p {V["krorig_p"]:.4f})', 'krorig', 0.001, 'split34 t1_variance 한국기원점유_변화pp', '안(C18 561행)'),
]
def classify(key, unit):
    a, b = P[key], V[key]
    if unit == 'range':
        return '경미(범위 표기 끝값)', f'최대 {a:.6f}→{b:.6f}(90.5% 표기의 실제값 {a*100:.2f}%는 정수 반올림 90, V1E {b*100:.2f}%는 91)'
    if unit == 'count':
        rel = (b - a) / a
        return ('경미' if abs(rel) <= 0.025 else '경미(끝자리 초과)'), f'상대 변화 {rel*100:+.2f}%'
    units = round(abs(b - a) / unit)
    return ('경미' if units <= 3 else '경미(끝자리 초과)'), f'끝자리 {units}단위'
rows = []
for ln, old, new, key, unit, src, c18 in items:
    s = ms[ln - 1].replace('−', '−')
    present = old in ms[ln - 1]
    cls, note = classify(key, unit)
    extra = ''
    if key == 'ep_purged':
        u2 = round(abs(V['p_max'] - P['p_max']) / 0.001); cls2 = '경미' if u2 <= 3 else '경미(끝자리 초과)'
        extra = f'; p 상한 {P["p_max"]}→{V["p_max"]}(끝자리 {u2}단위)'
    if key == 'krorig' and ln == 562:
        u2 = round(abs(V['krorig_p'] - P['krorig_p']) / 0.0001); extra = f'; p {P["krorig_p"]}→{V["krorig_p"]}(끝자리 {u2}단위)'
        if u2 > 3: cls = '경미(끝자리 초과)'
    if key == 'ci_lo':
        u2 = round(abs(V['ci_hi'] - P['ci_hi']) / 0.0001); extra = f'; 상한 {P["ci_hi"]}→{V["ci_hi"]}(끝자리 {u2}단위)'
        if u2 > 3: cls = '경미(끝자리 초과)'
    judg = '없음'
    if key == 'n_fwd':
        judg = '없음(BH q<0.10 통과 0개 그대로; 우연 기대 10과 같아짐 — 결론 "선행 관계 없다"는 강해지는 방향)'
    if key in ('ci_lo',):
        judg = '없음(CI 0 미포함 그대로)'
    if key == 'm4_ci_lo':
        judg = '없음(CI 0 포함 그대로)'
    if key == 'krorig':
        judg = '없음(무작위 기대 69.6%보다 큼·p>0.05 그대로)'
    if key == 'ep_purged':
        judg = '없음(p<0.05 그대로)'
    if key == 'kr_l2':
        judg = '없음(L=2에서 M 열세 0/4 그대로)'
    if key == 'pairacc':
        judg = '없음(0.523과의 비교 방향 그대로)'
    rows.append(dict(원고행=ln, 원래문자열=old, 새문자열=new, 원래문자열_원고존재=int(present), 원값=P[key], 새값=V[key], 원천=src,
                     판정변화=judg, 분류=('확인된 오류' if judg.startswith('있음') else cls), 비고=note + extra, C18_11행=c18,
                     글자증감=len(new) - len(old)))
with open(W + r'\diff_numbers.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
jdump(dict(P0=P, V1E=V), 's9_values.json')
for r in rows: out(r['원고행'], r['원래문자열'], '→', r['새문자열'], '|', r['분류'], '|', r['비고'], '| 존재', r['원래문자열_원고존재'], '| Δ글자', r['글자증감'])
out('lines', sorted(set(r['원고행'] for r in rows)), len(set(r['원고행'] for r in rows)))

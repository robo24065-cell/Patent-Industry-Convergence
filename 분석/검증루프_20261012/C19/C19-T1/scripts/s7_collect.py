# C19-T1 s7: sens_results.csv 모으기 + 사전 판정 적용
import sys, os, json, csv; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import C, W, out
TH = 0.25 * (0.47307183171311484 - 0.36580365189908404)
desc = {'P0': 'E40-auditD 최종 t0=2021 재현', 'VJ1': 'JP 공개일 없는 행 제외(모든 해, 150,332행)', 'VJ1s': '(보조) py 2021–22 JP 공개일 없는 행만 제외(54,958행)',
        'VJ2': 'JP 공개일 없는 행 공개연도 = 출원일+18개월', 'VJ3': '이중 수록 판정 JP 행만 제외(키1∪키2, 36,486행)',
        'VJ4a': '스펙 안 JP_* 6피처 = 0 고정', 'VJ4b': '스펙 안 JP_* 6피처 = 학습 평균 고정', 'VE1': 'EP 공백 달 등록일 결측(등록일이 공개연도를 정하는 EP 행 0 → P0와 같음, 미실행)',
        'VRNDJP0': '[위약 s=0] 쌍 아닌 공개일 있는 JP 행, py별 VJ1과 같은 수 무작위 제거', 'VRNDJP1': '[위약 s=1]', 'VRNDJP2': '[위약 s=2]',
        'VRNDJP7': '[verify_1 위약, 다른 코드 경로 RandomState(7), 짝 제외 안 함]', 'VNJP': '[verify_1 상한] JP 행 전체 제거'}
P = {}
for v in desc:
    f = os.path.join(C, f'paired_{v}.json')
    if os.path.exists(f): P[v] = json.load(open(f, encoding='utf-8'))
plc = [abs(P[v]['final']['delta']) for v in ('VRNDJP0', 'VRNDJP1', 'VRNDJP2')]
pmax = max(plc)
rows = []
sb0 = P['P0']['sb3']
for v, dsc in desc.items():
    if v == 'VE1':
        rows.append(dict(track='T3', variant=v, description=dsc, skill_final='', delta_vs_P0='0(구성상)', judgment='영향 없음(구성상)')); continue
    if v not in P: continue
    f = P[v]['final']; dv = P[v].get('dev') if isinstance(P[v].get('dev'), dict) else None; s3 = P[v]['sb3']; ex = P[v].get('exposed', {})
    d = f['delta']; dist = (abs(d) > pmax) and not (f['delta_ci95'][0] <= 0 <= f['delta_ci95'][1])
    m0_ok = f['skill_V_ci95'][0] > 0; sb_ok = s3['ci95'][0] > 0
    if v == 'P0':
        j = '공식 0.42289 재현(차 0), 예측 최대차 0; SB3 0.182 [0.124, 0.243] 재현'
    elif v.startswith('VRND') or v == 'VNJP':
        j = '참고(위약·상한, 판정 대상 아님)'
    else:
        if abs(d) >= TH or not m0_ok or not sb_ok: j = '경고'
        elif dist: j = '영향 없음(작지만 구분되는 변화)'
        else: j = '영향 없음'
        if f['skill_V'] < 0.42289 - TH: j += ' · 원고 오류 후보'
    rows.append(dict(track='T3', variant=v, description=dsc, skill_final=round(f['skill_V'], 6), delta_vs_P0=round(d, 6), delta_ci95_paired=[round(x, 5) for x in f['delta_ci95']],
                     skill_ci95_vs_M0=[round(x, 5) for x in f['skill_V_ci95']], n_test_cells=f['n_cells_V'], n_evt=f['n_evt_V'],
                     skill_vs_SB3=round(s3['skill_vs_sb3'], 5), skill_vs_SB3_ci95=[round(x, 5) for x in s3['ci95']],
                     dev_mean_skill=round(dv['skill_V'], 6) if dv else '', dev_delta=round(dv['delta'], 6) if dv else '', dev_delta_ci95=[round(x, 5) for x in dv['delta_ci95']] if dv else '',
                     exposed_med_abs_drank_hi=round(ex.get('med_abs_drank_hi', 0), 4), exposed_med_abs_drank_rest=round(ex.get('med_abs_drank_rest', 0), 4),
                     top10_hits_hi_P0_to_V=f"{ex.get('top10_hits_hi_P0')}→{ex.get('top10_hits_hi_V')}", top10_hits_all_P0_to_V=f"{ex.get('top10_hits_all_P0')}→{ex.get('top10_hits_all_V')}",
                     distinct_from_placebo=dist if v not in ('P0',) and not v.startswith('VRND') and v != 'VNJP' else '', judgment=j))
cols = ['track', 'variant', 'description', 'skill_final', 'delta_vs_P0', 'delta_ci95_paired', 'skill_ci95_vs_M0', 'n_test_cells', 'n_evt', 'skill_vs_SB3', 'skill_vs_SB3_ci95',
        'dev_mean_skill', 'dev_delta', 'dev_delta_ci95', 'exposed_med_abs_drank_hi', 'exposed_med_abs_drank_rest', 'top10_hits_hi_P0_to_V', 'top10_hits_all_P0_to_V', 'distinct_from_placebo', 'judgment']
with open(os.path.join(W, 'sens_results.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow({k: r.get(k, '') for k in cols}) for r in rows]
out('threshold', TH, 'placebo max', pmax)
for r in rows: out(r['variant'], r.get('skill_final'), r.get('delta_vs_P0'), r.get('delta_ci95_paired', ''), r.get('skill_vs_SB3', ''), r.get('skill_vs_SB3_ci95', ''), r.get('dev_delta', ''), r['judgment'])

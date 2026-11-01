# s6: feasibility.csv — P1–P4 판정표(PREREG §2 기준 그대로). 입력: dev_stats.json, p3_result.json, p4_result.json, essentiality_dev.csv
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4'
C = W + r'\cache'
st = json.load(open(C + r'\dev_stats.json', encoding='utf-8'))
p3 = json.load(open(C + r'\p3_result.json', encoding='utf-8'))
p4 = json.load(open(C + r'\p4_result.json', encoding='utf-8'))
P = pd.read_parquet(C + r'\dev_pos.parquet')
E = pd.read_csv(W + r'\essentiality_dev.csv')
rows = []
add = lambda crit, item, value, threshold, result, note='': rows.append(dict(criterion=crit, item=item, value=value, threshold=threshold, result=result, note=note))
for s in st:
    add('P1', f'pos_t0_{s["t0"]}', s['pos'], '>=200', 'pass' if s['pos'] >= 200 else 'fail',
        f'units={s["risk_units"]}; eligible={s["eligible_firms"]}; win_acc_rows={s["win_acc_rows"]}; mapped_prefix1={s["win_mapped_prefix1_rows"]}; event_rows={s["event_rows"]}')
tot = sum(s['pos'] for s in st)
uniq = P.drop_duplicates(['cd', 'k', 'rgst_no', 'd']).shape[0]
all200 = all(s['pos'] >= 200 for s in st)
add('P1', 'pos_sum_2012_2022', tot, '>=1000', 'pass' if tot >= 1000 else 'fail', '원점 합(PREREG 문언). 19개월 창 7개월 겹침으로 같은 사건이 두 원점에 들어감')
add('P1', 'pos_unique_events(진단)', uniq, '(참고)1000', 'diag_below' if uniq < 1000 else 'diag_ok', '겹침 제거 고유 (f,k,특허,날짜). 판정에는 쓰지 않음')
p1 = all200 or tot >= 1000
add('P1', 'P1_verdict', '', '모든 원점>=200 또는 합>=1000', 'pass' if p1 else 'fail', '원점별 기준은 전부 미달, 합 기준으로만 통과')
add('P2', 'N4_final_window', '2024-01-01~2025-12-31', '권리이전 결과·피처 아님', 'pass', 'final_spec_r4.json final.outcome_window(국세청 폐업일)')
add('P2', 'scored_tests_using_owner_2025plus', 0, '0', 'pass', 'T1 R3 acq <=2024-12-31, N4 피처 <=2023-12, T4 E<=2025-01-01, T5 레이블 2023-12 끝(verify_3 재확인)')
add('P2', 'P2_verdict', '', '', 'pass', '집계 수 열람은 exposure_log에 공개')
add('P3', 'cells_both_units_share', p3['cells_both_units_share'], '>=0.95', 'pass' if p3['cells_both_units_share'] >= 0.95 else 'fail', f'{p3["cells_both_units"]}/{p3["cells_total"]} 칸(KSIC×원점), 위험집합 단위 기준')
add('P3', 'cells_both_pos_share(보조)', p3['cells_both_pos_share_of_pos_cells'], '보고', 'report', f'양성 있는 칸 {p3["cells_with_pos"]}개 중')
add('P3', 'map_rate_all', p3['map_rate_all'], '보고', 'report', json.dumps(p3['map_share'], ensure_ascii=False))
add('P3', 'map_rate_corp', p3['map_rate_corp'], '<0.5 경고', 'warn' if p3['map_rate_corp'] < 0.5 else 'ok', '')
for k, v in p3['smd_mapped_vs_unmapped'].items():
    add('P3', f'SMD_mapped_vs_unmapped_{k}', v, '|SMD|>0.25 경고', 'warn' if abs(v) > 0.25 else 'ok', '')
add('P3', 'KSIC_Duncan_mapped_vs_unmapped', p3['ksic_duncan_mapped_vs_unmapped'], '보고', 'report', f'법인 표지만 {p3["ksic_duncan_corp_only"]}')
add('P3', 'pos_rate_ratio_NPSlinked_over_unlinked', p3['pos_rate_ratio_linked_over_unlinked'], '0.5~2 밖 경고', 'warn' if not (0.5 <= p3['pos_rate_ratio_linked_over_unlinked'] <= 2) else 'ok',
    f'연결 {p3["pos_rate_linked"]} / 비연결 {p3["pos_rate_unlinked"]}; 원점별 {json.dumps(p3["pos_rate_ratio_by_t0"])}')
add('P3', 'P3_verdict', '', '칸 공존>=0.95', 'pass' if p3['P3_decision_cells_both_ge_0.95'] else 'fail', '경고 3건: ' + '; '.join(p3['P3_warnings']))
a = p4['all']
add('P4', 'P4a_med(2026-05~07)/med(2026-01~04)', a['P4a_ratio'], '>=0.70', 'pass' if a['P4a_pass'] else 'fail', f'{a["P4a_med_2026_05_07"]} / {a["P4a_med_2026_01_04"]}')
add('P4', 'P4b_sum(2026-01~07)/sum(2025-01~07)', a['P4b_ratio'], '>=0.70', 'pass' if a['P4b_pass'] else 'fail', f'{a["P4b_sum_2026_01_07"]} / {a["P4b_sum_2025_01_07"]}')
add('P4', 'P4c_sum(2026-01~07)/mean(2022-24 1~7월)', a['P4c_ratio'], '>=0.70', 'pass' if a['P4c_pass'] else 'fail', f'{a["P4b_sum_2026_01_07"]} / {a["P4c_mean_2022_24_01_07"]}')
p4ok = a['P4a_pass'] and a['P4b_pass'] and a['P4c_pass']
add('P4', 'P4_verdict', '', 'a,b,c 모두', 'pass' if p4ok else 'fail', 'P4-b 실패')
for col in ['no_bulk', 'corp', 'stock_le2024', 'patent_date_events']:
    x = p4[col]
    add('P4진단', f'{col}_a/b/c', f'{x["P4a_ratio"]}/{x["P4b_ratio"]}/{x["P4c_ratio"]}', '(판정 외)', '/'.join('p' if x[k] else 'f' for k in ['P4a_pass', 'P4b_pass', 'P4c_pass']), '')
pool = E[E.t0 == 'pooled_2012_2022']
for r in pool.itertuples():
    add('필수성(가)', f'jaccard_{r.assign}', r.jaccard, '<0.5', 'support' if r.jaccard < 0.5 else 'insufficient', f'재현율 {r.recall_of_provided}; 공개 사건 {r.events_public} / 제공 {r.events_provided} / 교집합 {r.intersect}')
ess = all(pool.jaccard < 0.5)
verdict = 'reject' if not (p1 and p4ok and p3['P3_decision_cells_both_ge_0.95']) else 'pass'
add('종합', 'C-D_verdict', verdict, 'P1-P4 모두 통과', verdict, f'필수성 {"지지" if ess else "불충분"}; PREREG_final 미작성' if verdict == 'reject' else '')
pd.DataFrame(rows).to_csv(W + r'\feasibility.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string())

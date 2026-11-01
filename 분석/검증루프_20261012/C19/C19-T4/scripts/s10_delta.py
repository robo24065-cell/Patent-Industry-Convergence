# -*- coding: utf-8 -*-
# C19-T4 s10: (1) v20 반영으로 상태가 바뀌는 상태표 행(C18-T3 build_plan O1–O9·A1–A10 대응 ID) 목록, (2) 제안 출처 칸 글자 수.
import csv, os, json
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.dirname(os.path.dirname(D))
R = os.path.join(os.path.dirname(B), '보고서_20261012')
st = {r['ID']: r for r in csv.DictReader(open(os.path.join(B, 'C18', 'C18-T3', 'errata_status_v18.csv'), encoding='utf-8-sig'))}
V20 = open(os.path.join(R, 'manuscript_v3.md'), encoding='utf-8').read().split('\n')
mp = {'E-C17T3-01': 'O1(권장판 — 반영 행의 하위 판)', 'E-C16T2-06': 'O2', 'E-C17T3-04': 'O3', 'E-C16T2-04': 'O4', 'E-C17T1-01b': 'O5', 'E-C16T3-S': 'O6',
      'E-C13T2-02': 'O7', 'E-C09T2-04': 'O8', 'E-C13T1-01': 'O9', 'E-C03T3-11': 'A1', 'E-C03T3-10': 'A2', 'E-C16T4-02': 'A3', 'E-C02T1-07': 'A4',
      'E-C15T1-03': 'A5', 'E-C17T3-B2': 'A6', 'E-C16T4-03': 'A7', 'E-C02-T3-12': 'A8', 'E-C07T1-08': 'A9', 'E-C14T2-05': 'A10',
      'E-C15T1-02': 'A3(655행 출처)', 'E-C15T4-A1': 'A3(482행 출처)', 'E-C01-X2': 'A1(C-LAND-prov 절 열; CIRC-BOUND 3.6.2, 4.4는 남음)'}
rows = []
for k, v in mp.items():
    s = st.get(k)
    rows.append({'ID': k, 'C18_v19상태': s['상태'] if s else '(표에 없음)', 'C18_분류': s['분류'] if s else '', 'v20반영경로': v,
                 'v20상태(제안)': '반영됨' if k not in ('E-C01-X2',) else '부분→반영됨(검토)'})
with open(os.path.join(D, 'status_v20_delta.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
adds = {524: '; (분석)\\검증루프_20261012\\C12\\C12-T1\\sens_results.csv, verify_0\\v1_agg_off_*.json',
        553: '; (분석)\\검증루프_20261012\\C11\\C11-T2\\res\\s7_landscape.csv',
        647: '; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s3_rho_table.csv',
        648: '; (분석)\\검증루프_20261012\\C10\\C10-T3\\res\\s3_rho_table.csv',
        482: '; (분석)\\검증루프_20261012\\C11\\C11-T2\\res\\s7_landscape.csv'}
info = {}
for ln, a in adds.items():
    line = V20[ln - 1]
    info[ln] = {'id': line.split('|')[1].strip(), 'chars_add': len(a), 'current_source': line.rstrip().rstrip('|').split('|')[-1].strip()[:200]}
cnt = {}
for r in rows:
    cnt[r['C18_v19상태']] = cnt.get(r['C18_v19상태'], 0) + 1
json.dump({'delta_n': len(rows), 'by_v19_status': cnt, 'source_adds': info}, open(os.path.join(D, 'cache', 's10_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

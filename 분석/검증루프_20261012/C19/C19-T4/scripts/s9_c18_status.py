# -*- coding: utf-8 -*-
# C19-T4 s9: C18 ERRATA 항목 상태표(v19 기준 판정 + v20 상태). 근거는 cache\s6_c18.json(기계 검색)·s5·직접 대조 결과.
import csv, os, json
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s6 = {r['item']: r for r in json.load(open(os.path.join(D, 'cache', 's6_c18.json'), encoding='utf-8'))}
def ln(k, f):
    return ';'.join(map(str, s6[k].get(f, []))) if k in s6 else ''
rows = []
def add(ID, sub, scope, g, j19, v20, regrade, ev19, ev20, num, memo):
    rows.append({'ID': ID, '하위': sub, '범위': scope, '원등급': g, 'v19판정': j19, 'v20상태': v20, '재등급': regrade,
                 'v19근거줄': ev19, 'v20근거줄': ev20, '숫자대조': num, '메모': memo})
add('E-C18T1-01', '권장판(+11)', '본문 410행', 'warning·권장', '미반영-여전히 필요', '반영(권장판; v20 md·docx·pdf 모두 있음)', '권장',
    ln('E-C18T1-01', 'old_v19'), ln('E-C18T1-01', 'new_v20'),
    '일치: monthly_capture.csv 연 합계 KR/O_pat 비율을 2023년 비율 0.2602 기준으로 보면 2024 −19.5%, 2025-01~07 −47.3%(2021–23 평균 기준이면 −19.0%·−46.9%, 기준 의존 ±0.5%p); 공백 달 2024-01~08·2025-02~08 일치',
    'v19는 원인·19%는 맞으나 더 큰 2025년 결손(T1 최종 창 누락의 70%)을 빠뜨림 → 한계 서술 불완전(권장). C18 머리말의 "필수 0·권장 0 → v19 그대로 제출 가능"은 C01–C17 상태표 기준이고 같은 C18 절의 이 권장 항목은 v19에 열려 있었다 — v20에서 해소')
O = [('O1', '409행 신설 분류 셀', '0.38=C17-T3 V1r 0.3802, 0.42=C05 0.4163(C18-T3 recheck 일치)'),
     ('O2', '143행 광공업지수', '숫자 없음'), ('O3a', '143행 법인 연결표·DART 업종', '숫자 없음'), ('O3b', '143행 국민연금은', '숫자 없음'),
     ('O4', '410행 N6 미연결 기업은 결과 0', '숫자 없음(N6 DATA_NOTES 근거)'), ('O5', '410행 T1 법인·위험쌍', '숫자 없음'),
     ('O6', '96행 표 3 실용신안 포함', '9.34%·10.95% 분모 실용신안 포함(C18-T3 recheck 표3-③, kr_prefix.csv)'),
     ('O7a', '130행 0 밑인 T4(−0.003)', '−0.003(robustness_table.md; C13-T2 시드 하한 모두 <0)'), ('O7b', '409행 0 밑이고(−0.003)', '같음'),
     ('O8', '142행 약 47–48%', '원천 46.87–48.21%(C03-T3 _s8_unseen.json) — 사사오입 47–48'),
     ('O9', '286행 예고등록(직권)', '숫자 없음(등록령 제6조①4·제19조②, C13-T1)')]
for k, sub, num in O:
    add('E-C18T3-O', k + ' ' + sub, '본문', '선택', '미반영-여전히 필요', '반영(v20 md·docx·pdf)', '선택', ln(k, 'old_v19'), ln(k, 'new_v20'), num,
        'v19 대비 v20 = C18-T3 manuscript_v19_option.md + E-C18T1-01 권장판 + C18-T1-REGGAP 행(diff 확인, 그 밖 차이 0)')
add('E-C18T3-A', 'A1–A10(부록 절 열·출처 칸)', '부록(docx 밖)', '선택', '미반영-여전히 필요', '반영', '선택',
    'CIRC-AGREE-CC 496, LINK-NPS 509, NTIS-SRC 658, C-LAND-prov 480', '496;509;659;480; pkr_ipc_altA 498·499·501; s4_horizon 656',
    '숫자 변경 0', 'v20 = option 판(부록 hunk 그대로). 남은 출처 칸 공백: 524 OFF-N5-SKILL, 553 C-LAND-txt_F, 647·648 C-RCA-TXT/FREE-IPC(이 과제 proposed_errata P1)')
add('E-C18T1-B1', 'C18-T1-REGGAP 행', '부록', '선택', '미반영-여전히 필요', '반영(v20 539행)', '선택', '', ln('B1', 'new_v20'),
    '일치: −19.5%·−47.3%(위와 같음), Δskill −0.0021560 [−0.004138, −0.000351] → −0.0022 [−0.0041, −0.0004](C18-T1 sens_results.csv S1)', '')
add('E-C18T3-F', 'facts_patch_v19 수정 후 적용', 'facts', '조건부', '미반영-여전히 필요', '미반영(facts.csv d8d3c2d6·supplement c7fb7d78·R12 1b6fd7c4 그대로)', 'facts',
    '', '', '해당 없음', 'C14–C17 facts 항목 8개(이번 표본)도 이 항목으로 통합돼 있다. 적용 전 결함 3가지(LINK-IO 잘림, 추가 21행 칸 부족, 부록 베낀 18행 근거) 수정 필요 — 원 판정 유지')
for ID, what in [('E-C18T1-D1', 'ERRATA E-C17T2-D2 (iv) 문서 정정'), ('E-C18T1-D2', '데이터 노트(수정판)'), ('E-C18T1-D3', 'FINDINGS 주의(C02-T2 −24%)'),
                 ('E-C18T2-D1', '_shared 데이터 노트(제공 메타 정제 규칙)'), ('E-C18T2-D2', 'FINDINGS C17-T4 옆 정정 표시'), ('E-C18T2-D3', '후속 과제(T3 JP 노출·T4 fam_n)'),
                 ('E-C18T4-D1', '설계 교훈'), ('E-C18T3-G', 'ERRATA 문서 정정(E-C10T4-05 +18, E-C13T1-01 = O9)')]:
    add(ID, what, '원고 밖', '원고 밖', '미반영-여전히 필요', '미반영(원고 밖 — 대상 문서 수정 여부 확인 안 함, PREREG 규칙)', '원고 밖', '', '', '해당 없음',
        'E-C18T3-G 후반(E-C13T1-01 = O9)은 v20 O9 반영으로 원고 쪽 해소' if ID == 'E-C18T3-G' else '')
add('E-C18T2-01~05', '이중 수록 제거 끝자리 정정(반박으로 빠짐)', '본문', '반박', '폐기 타당', '현행 유지(v20 해당 줄 숫자 변경 없음)', '',
    '71·97·142·191·496·517·543·550·551·559·561', '같음', '현재 숫자는 제공 데이터 그대로 계산값과 일치(C18-T2·T3)', '11행+facts 7항목 묶음 확정 전 보류가 맞다. 대안 B 주석은 선택')
add('C18-T4-407', '4.4-1 근거 B "KSIC 없이" 단서(반박)', '본문', '반박', '폐기 타당', '없음(v20 "KSIC 없이" 0건)', '', '', '', '해당 없음', '')
add('E-C18T1-D2 원판', '"공란은 개인 출원에는 거의 없다"(반박)', '원고 밖', '반박', '폐기 타당', '원고 무관(v20 0건)', '', '', '', '해당 없음', '수정판 E-C18T1-D2가 대체')
add('C18-T3 facts_patch_v19 그대로 적용', '(반박)', 'facts', '반박', '폐기 타당', 'facts 원본 그대로', '', '', '', '해당 없음', 'E-C18T3-F(조건부)로 대체')
with open(os.path.join(D, 'c18_items_status.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows))

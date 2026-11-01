# C16-T4 s2: C15 ERRATA 항목별 반영 상태 (현재 원고 4cbf3d55 · facts · 원장 · 트랙 문서 · 루트 USER_REQUESTS 실측)
import csv, sys, os, re
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; L = AN + r"\검증루프_20261012"; MC = AN + r"\모델경쟁_20260927"
OUT = L + r"\C16\C16-T4"
ms = open(R + r"\manuscript_v3.md", encoding='utf-8').read().replace('\r\n', '\n').split('\n')
def ln(n): return ms[n - 1]
facts = open(R + r"\facts\facts.csv", encoding='utf-8-sig').read()
sup = open(R + r"\facts\facts_supplement.csv", encoding='utf-8-sig').read() + open(R + r"\facts\facts_supplement_R12.csv", encoding='utf-8-sig').read()
FIND = open(L + r"\FINDINGS.txt", encoding='utf-8').read()
t4notes = open(MC + r"\T4_특허포기예측\DATA_NOTES.md", encoding='utf-8').read()
ur_root = os.path.exists(L + r"\USER_REQUESTS.txt")
E = open(L + r"\ERRATA.txt", encoding='utf-8').read()
rec15 = E[E.find('[메인 세션 반영 기록 2026-10-17 (15)]'):]

def has(n, s): return s in ln(n)
items = []
def add(eid, where, status, evidence, main_claim, note=''):
    items.append(dict(ID=eid, 위치=where, 상태=status, 실측근거=evidence, 메인기록15=main_claim, 비고=note))
def claim(eid):
    seg = rec15.split('\n')
    base = eid[:-3]; suf = eid[-2:]
    for s in seg:
        if s.startswith('반영됨:') and (eid in s or (base + '-01·' + suf) in s): return '반영됨'
        if s.startswith('시도 후 되돌림') and eid in s: return '되돌림'
        if s.startswith('미반영') and eid in s: return '미반영'
    if re.search(r'-F\d*$', eid): return '미반영(facts C 항목으로 일괄)'
    if re.search(r'-D\d+$', eid): return '미반영(원고 밖 D 항목으로 일괄)'
    return '언급 없음'

# A. 본문
a = has(109, '라운드마다 새로 잠갔고,') and has(109, 'T1·T3 공식 창은 앞 라운드의 다른 정의 최종시험(T1 0.418, T3 0.392)과 결과 기간이 겹친다.')
b = has(50, '라운드마다 새로 잠근 최종시험 1회')
add('E-C15T4-01', '109행(01b)+50행(표 1)', '반영' if a and b else ('부분' if a or b else '미반영'),
    f"109행 01b 문구 {'있음' if a else '없음'}; 50행 표 1 '라운드마다 새로 잠근' {'있음' if b else '없음'}; '홀드아웃 재사용' {'없음' if '홀드아웃 재사용' not in ln(109) else '있음'}", claim('E-C15T4-01'),
    '01a 대신 01b. 50행은 시험 빌드 없었으나 v17 실측 20쪽(5절)')
add('E-C15T4-02', '143행', '반영' if has(143, '공식 통계 대비 건수 비율은 KR 등록 2012–23년 0.25–0.26, CN 2018–22년 0.20–0.21로 안정적이다.') else '미반영',
    '143행 F2z(−2자) 문구 일치; 원천 KR 0.253–0.263·CN 0.201–0.207 재계산 일치(numeric_checks.csv)', claim('E-C15T4-02'))
c3 = [has(229, 'KIPRIS 권리목록(범위밖 특허·디자인·실용신안), ECOS'), has(99, 'KIPRIS 권리목록(범위밖 특허·디자인·실용신안)은 T1 피처로'), has(251, 'KIPRIS Plus 권리목록·디자인·실용신안')]
add('E-C15T4-03', '229·99·251행', '반영' if all(c3) else ('부분' if any(c3) else '미반영'), f'(a)229 {c3[0]} (b)99 {c3[1]} (c)251 {c3[2]}', claim('E-C15T4-03'), "'범위밖 피처가 성적을 부풀렸다' 류 문구 없음")
add('E-C15T4-04', '265행', '반영(04b)' if has(265, '(원판 챔피언 AUC 0.899는 더 높다)') else ('부분(04a)' if has(265, '두 원판 대비 우위') else '미반영'),
    '04b 문구 일치; 0.8993→0.899 재계산 일치', claim('E-C15T4-04'))
add('E-C15T4-05', '191행', '반영' if has(191, '합계 추세는 연계표로 읽히지만 한국 지표는(US·JP·EP도) 그렇지 않다.') else '미반영',
    '원안(+11자) 일치; KR 0.561·US 0.551·JP 0.596·EP 0.563·합 0.291 원천 일치', claim('E-C15T4-05'))
add('E-C15T4-06', '362행', '반영' if has(362, '예측 피처(T3 경계·N9·N6)') else '미반영', '원안(+3자) 일치', claim('E-C15T4-06'))
add('E-C15T4-07', '79행', '반영' if has(79, '제공 범위 KR 등록특허·실용신안 656,360건을') else '미반영', 'registry_full.sqlite(mode=ro) 10: 615,042·20: 41,318 = 656,360', claim('E-C15T4-07'),
    "추가 확인 필요로 남긴 '출원 분모 1,087,511의 실용신안 포함 여부'는 미해결")
t11 = ln(388)
add('E-C15T1-01', '388행(표 11)', '미반영(되돌림)' if '역전' not in t11 else '반영', "388행에 '역전'·'6년차' 없음; 부록 651행에만 있음", claim('E-C15T1-01'),
    'B안(+8자)은 21쪽이 되어 되돌림(메인 기록). 본문 4.2의 창 의존 단서는 제출본에 없음 → 경고 잔존')
# B. 부록
add('E-C15T3-01', '519행', '반영' if has(519, 'hyb_pkr(진단) 0.7755(등록번호 연결·C15-T3)') else '미반영', '0.77547→0.7755 일치(s5_t3_summary.json)', claim('E-C15T3-01'),
    '출처 칸은 여전히 (MC)\\c5_R10\\c5_t3_summary.json(0.7755·0.7626·0.7605 없음) — E-C12T2-08 미반영과 같은 성격')
add('E-C15T3-02', '638행', '반영' if has(638, 'hyb_pkr 67%(등록번호 연결·C15-T3)') else '미반영', '보존율 0.668→67% 일치', claim('E-C15T3-02'), '출처 칸 동일 문제')
add('E-C15T4-07b', '503행', '반영' if has(503, '제공 범위 KR 등록 656,360건(특허 615,042·실용신안 41,318)') else '미반영', 'sqlite 재집계 일치', claim('E-C15T4-07b'))
l534 = open(L + r"\C15\C15-T2\cache\l534_new.txt", encoding='utf-8').read().replace('\r\n', '\n').split('\n')[1].strip()
add('E-C15T2-01', '534행', '반영' if ln(534).strip() == l534 else '미반영', 'l534_new.txt 2행과 문자 동일; 76/76·0.376–0.384·0.124–0.143·0.06–0.19·0.423–0.432 재계산 일치', claim('E-C15T2-01'),
    "verify 권고 'SE=CI폭/3.92' 정의 병기는 없음(E-C15T2-02 조건부 권고)")
p2 = [has(488, '(다른 학습 시드 10묶음: 0.472–0.475, 하한 0.453–0.456, C15-T2)'), has(485, '(다른 학습 시드 10묶음: 0.423–0.432, 공식이 최솟값, C15-T2)'),
      has(577, '다른 학습 시드 10묶음: 29.8–31.6%, 공식은 두 번째로 높음, C15-T2)')]
add('E-C15T2-02', '488·485·577행', '반영' if all(p2) else ('부분' if any(p2) else '미반영'), f'488 {p2[0]} 485 {p2[1]} 577 {p2[2]}; 본문 233·385행 공식값 그대로', claim('E-C15T2-02'),
    "SE 정의 병기는 배율을 적을 때만 요구 — 485·488·577행은 배율 없음(해당 없음)")
add('E-C15T2-03', '483행', '반영' if has(483, 'T3 −0.005(학습 시드 10묶음 −0.005~+0.003)') else '미반영', '−0.00522·+0.00307 재계산 일치', claim('E-C15T2-03'))
v = has(651, '; 7년 내 37.7% 대 35.2%, 10년 내(등록 2007–15) 55.5% 대 52.2%(6년차 기한 집중, C15-T1)'); s = 'C15-T1\\cache\\s4_horizon.csv' in ln(651)
add('E-C15T1-02', '651행', '반영' if v and s else ('부분' if v else '미반영'), f"값 칸 {'있음' if v else '없음'}, 출처 칸 s4_horizon.csv {'있음' if s else '없음'}", claim('E-C15T1-02'),
    '값은 소멸등록일(regdate) 기준 재계산 일치. 출처 칸만 빠짐')
add('E-C15T1-03', '515행', '반영' if '실용신안 포함; 특허 불납 293,169' in ln(515) else '미반영', "515행 '소멸등록 349,507' 그대로", claim('E-C15T1-03'), '선택·최하위')
v = has(482, '(최빈 동률 규칙 의존: 표 동률만 반대 0.473, 배정까지 반대 0.476 → 본문 168행 최대 0.48)'); s = 'v5_frac.json' in ln(482)
add('E-C15T4-A1', '482행', '반영' if v and s else ('부분' if v else '미반영'), f"값 칸 {'있음' if v else '없음'}, 출처 칸 v5_frac.json {'있음' if s else '없음'}", claim('E-C15T4-A1'),
    '0.4727→0.473, 0.4755→0.476 재계산 일치. 0.476의 실제 원천은 v5_out.json range_168.REG_tie_desc_full(v5_frac.json에는 0.473·0.469만 있음)')
# C. facts
add('E-C15T3-F', 'facts.csv 151·160·163·166·170·171', '반영' if '0.7755' in facts else '미반영', "facts.csv에 0.7755 없음(파일 수정일 10-12)", claim('E-C15T3-F'), '이번 바퀴 facts_patch_consolidated.csv에 포함')
add('E-C15T4-F', 'facts.csv DATA-KR-REG', '반영' if '615,042' in facts else '미반영', 'facts.csv에 615,042 없음', claim('E-C15T4-F'), '패치에 포함')
add('E-C15T1-F1', 'facts_supplement C15-T4-CAPTURE', '반영' if 'C15-T4-CAPTURE' in sup else '미반영', 'supplement에 행 없음', claim('E-C15T1-F1'), '패치에 포함')
add('E-C15T1-F2', 'facts_supplement C15-NTIS-HORIZON', '반영' if 'C15-NTIS-HORIZON' in sup else '미반영', 'supplement에 행 없음', claim('E-C15T1-F2'), '패치에 포함')
add('E-C15T2-F1', 'facts_supplement C15-SEED-T1T3', '반영' if 'C15-SEED-T1T3' in sup else '미반영', 'supplement에 행 없음', claim('E-C15T2-F1'), '패치에 포함')
# D. 원고 밖
add('E-C15T1-D1', 'FINDINGS(원장 정정)', '반영(원장 정정 줄)' if '327–334일대는 2010–2016 전체' in FIND else '미반영', 'FINDINGS 369행 [C15][C15-T1][warning] (원장 정정) 줄 존재; 24행 원문은 append-only라 그대로', claim('E-C15T1-D1'))
add('E-C15T1-D2', 'T4 DATA_NOTES', '반영' if '1.054' in t4notes else '미반영', "T4_특허포기예측\\DATA_NOTES.md에 '1.054'·'C15-T1' 없음", claim('E-C15T1-D2'), '트랙 문서는 이 작업에서 쓰기 금지')
add('E-C15T1-D3', '설계 규칙', '해당 바퀴 확인 불가', '다음 PREREG에 적용할 규칙(문서 반영 대상 아님)', claim('E-C15T1-D3'))
add('E-C15T2-D1', 'FINDINGS(원장 정정)', '반영(원장 정정 줄)' if '9개 트랙 모두 시드 묶음 시험을 거쳤다' in FIND else '미반영', 'FINDINGS 374행 [C15][C15-T2][warning] (원장 정정) 줄 존재', claim('E-C15T2-D1'))
add('E-C15T2-D2', '운영', '해당 없음(운영 지침)', '연례 재학습 시 적용', claim('E-C15T2-D2'))
add('E-C15T3-D1', '운영', '해당 없음(운영 지침)', '재현 점검 규칙', claim('E-C15T3-D1'))
add('E-C15T4-D1', '루트 USER_REQUESTS.txt', '미반영→이번 바퀴 작성' if not ur_root else '반영', f"작업 전 루트 USER_REQUESTS.txt {'없음' if not ur_root else '있음'}", claim('E-C15T4-D1'), 'C16-T4가 UR-19 포함·제공처 기준 합침으로 작성(user_requests_final.txt)')
with open(OUT + r"\reflected_C15.csv", 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(items[0].keys())); w.writeheader(); w.writerows(items)
from collections import Counter
print(Counter(i['상태'] for i in items))
for i in items: print(i['ID'], i['상태'], '|', i['메인기록15'])

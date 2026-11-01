# C16-T4 s7: change_audit_raw.csv(s1) + 반박 반영 판단 + 시험 빌드 결과 → change_audit.csv(최종)
import csv, sys, json
sys.stdout.reconfigure(encoding='utf-8')
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C16\C16-T4"
raw = list(csv.DictReader(open(OUT + r"\change_audit_raw.csv", encoding='utf-8-sig')))
tb = json.load(open(OUT + r"\build_test\test_builds.json", encoding='utf-8'))
REF = {  # 반박(verify) 결과 대조
 50: '반박 대상 아님(메인 판단 항목). C16 N0 재빌드 = v17 document.xml 동일, 20쪽',
 79: "C15-T4 verify_0 U003-b(실용신안 포함) 반영. C13T3-05 과장 정정(E-C14T2-02)이 반영됐고, 금지 표현은 없음",
 99: "C15-T4 verify U039·U020: '단서 누락' 유지 → 권리목록 명시. '범위밖 피처가 성적을 부풀렸다' 류 표현 없음(C14-T3 민감도와 맞음)",
 109: "C15-T4 verify_0·1 U007 '과장' 유지(모순 아님) → 01b 반영. '홀드아웃 재사용' 같은 표현 없음. 앞 라운드 0.418·0.392는 R2_결과 원천 일치",
 143: "C15-T4 verify U019 → 기간 조건(KR 2012–23·CN 2018–22) 들어감. C14-T3 '상한이라 쓰지 말 것' 지켜짐('영향 0.007 이하')",
 191: "C15-T4 verify U031 '과장 아닌 단서 누락' → US·JP·EP 병기. KR 0.561·US 0.551·JP 0.596·EP 0.563 원천 일치",
 229: "U039·U020 반영. C14-T3-03 정의 정밀화 반영",
 251: "U039 그림 6 캡션 반영",
 265: "C15-T4 verify U042 '두 원판 대비로 한정하면 참' → 04b(챔피언 0.899 병기). 0.8993 원천 일치",
 334: "C14-T1 반박 실패 결과(법인×대상 업종 모든 추정량 0 포함) 반영. 13·354·362·407·750행 '이원 군집 CI 0 포함' 단서 유지 확인",
 362: "C15-T4 verify U086 '보수 방향 단서 누락' → 'T3 경계' 반영",
 482: "C11-T2 verify_5 반박(동률 처리 의존) 반영. 0.476의 실제 원천은 v5_out.json range_168.REG_tie_desc_full.max_raw=0.4755(ERRATA가 적은 v5_frac.json에는 0.473·0.469만 있음)",
 483: "C15-T2 verify_0·1 반박 실패. 원천 일치",
 485: "C15-T2 verify_1: 공식 b0가 10묶음 중 최솟값(순위 10/10) → '공식이 최솟값' 일치",
 488: "C15-T2 verify_0(v1) 20개 CI 재계산 일치",
 498: "C14-T2 독립 재현 71.899%와 일치(반박 실패)",
 499: "C14-T2 독립 재현 73.761%와 일치",
 501: "C14-T2 독립 재현 72.821%와 일치",
 503: "C15-T4 verify_0 U003-b(registry sqlite) 일치. C16 재집계도 일치",
 519: "C15-T3 verify_0: 독립 9원점 재학습 0.77550 → 0.7755, 넷째 자리는 빌드 잡음(±0.0002). R8 결함 경로 숫자(0.7735·'미재계산') 원고에 0회",
 534: "C15-T2 verify_0 v1 반박 실패(76/76, 범위 일치). 정보용 지적 두 가지는 미반영(경미): ID 'C13-SEED6'에 C15 두 트랙을 담은 명칭 불일치, 'SE=CI폭/3.92' 정의 미병기",
 542: "C14-T3 분모(목록 고유 2,047,575) 확인",
 577: "C15-T2 verify_1: T1 상위 1000 313은 2/10위 → '두 번째로 높음' 일치",
 638: "C15-T3 verify_0 보존율 66.9% → 67% 일치. '61%'·'미재계산' 원고에 0회",
 651: "C15-T1 verify_0 R5: 소멸등록일 기준 7·10년에서도 역전 → 부록 값은 regdate 열과 일치. 같은 반박 결과가 본문 388행(표 11)에는 없음(B안 21쪽으로 되돌림) — C16 시험 빌드 B1(+5자)·B2(+1)·B3(−3) 모두 20쪽",
}
rows = []
for r in raw:
    ln = int(r['행'])
    num = r['숫자근거']
    nchk = num.count('[') if num else 0
    bad = '(X)' in num
    if r['영역'] == '본문':
        build = r['빌드대조'] + ('' if ln != 50 else f"; C16 N0 재빌드: v17 document.xml 동일={tb['N0']['docxml_same_as_v17']}, {tb['N0']['pages']}쪽, 문단 {tb['N0']['n_para']}")
    else:
        build = r['빌드대조']
    verdict = r['판정']
    if '출처 칸 보충 없음' in verdict:
        cls = '일치(숫자)·원안 반영·출처 칸 보충 누락(warning)'
    elif bad:
        cls = '불일치'
    elif r['문구누락'] or r['금지문구']:
        cls = '원안 미반영'
    else:
        cls = '일치·원안 반영' if nchk else ('원안 반영(시험 빌드 동일)' if r['영역'] == '본문' and '동일' in r['빌드대조'] else '원안 반영')
    rows.append(dict(행=ln, 영역=r['영역'], 바뀐판=r['바뀐판'].replace('v16', 'v16(C14 반영)').replace('v17', 'v17(C15 반영)'), ERRATA=r['ERRATA'],
                     판정=cls, 원안대조=('원안 문구 포함' if not r['문구누락'] else '누락: ' + r['문구누락']) + ('' if not r['금지문구'] else '; 금지/옛 문구: ' + r['금지문구']),
                     반박반영=REF[ln], 빌드대조=build, 숫자검산=(f"{nchk}개 재계산 모두 일치" if nchk and not bad else ('불일치 있음' if bad else '숫자 변경 없음(문구)')), 근거=num))
with open(OUT + r"\change_audit.csv", 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter(r['판정'] for r in rows))

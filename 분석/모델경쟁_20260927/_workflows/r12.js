export const meta = {
  name: 'r12-confirm-and-revise',
  description: '12라운드: 근거 A 확인 시험(공개 최강 FREE-IPC 대비, 사전 등록) + 그림 결함 수정 → 원고 수정판',
  phases: [
    { title: '확인·그림', detail: '근거 A 확인 시험, 그림 수정' },
    { title: '수정', detail: '원고 v2' },
  ],
}

const B = String.raw`C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`
const R = B + String.raw`\분석\모델경쟁_20260927`
const OUT = B + String.raw`\분석\보고서_20261012`

const COMMON = `
[배경] 「특허-산업 융합데이터 활용 아이디어 공모전」 보고서(20쪽, 제출 2026-11-13) 마무리 단계. 사용자 기준: 제공 특허-산업 융합데이터가 필수라는 것을 숫자로, 계속 검증.
[현재 원고] ${OUT}\\manuscript_v1_from_response.md (편집 메모 §C 그림 조치, §D 남은 확인, §E 쪽수 포함). 숫자 고정표 ${OUT}\\facts\\facts.csv, facts_supplement.csv. 그림 ${OUT}\\figs\\ (make_data_figs.py, make_diagrams.py, _style.py).
[11라운드 검증 요약] ${OUT}\\verify_R11_summary.md — 근거 C는 '부분'(조건부 유지). 사후 분석: KR 문서에 FREE-IPC(해설서 단독 텍스트 라벨 + IPC 최빈 평활)를 쓰면 기업 5자리 업종 일치에서 제공 − FREE-IPC = NPS +0.61%p [−0.26, +1.55], DART +6.02%p [+4.47, +7.59]; 시간 분리판 NPS +1.07%p, DART +9.29%p. 즉 근거 A의 "공개 배정 대비 +10%p/+25%p"는 TXT-FREE 대비 값이고, 공개 최강 대비로는 DART만 남는다. 폴더: ${R}\\_shared_report\\label_origin_R10\\foreign_public_R11\\
[절대 규칙] 원자료·트랙 폴더 읽기만. 지정 폴더에만 쓴다(.md 쓰기가 막히면 응답으로). 공식 성적 변경 금지. 불리한 결과도 그대로. 메모리 4GB, 스크립트 9분 이내. 한국어.
`

phase('확인·그림')
const [conf, figs] = await parallel([
  () => agent(`${COMMON}
[임무 — 근거 A 확인 시험(사전 등록)] 폴더 ${R}\\_shared_report\\label_origin_R10\\confirm_R12\\ (만들어서 여기에만).
**실행 전에 PREREG.txt**에 판정 기준을 적어라(예: 주 지표 = 기업 평균 5자리 일치, 비교 = 제공 vs FREE-IPC(시간 분리판: 표는 ≤2014 문서로만, 2015–19 문서 평가), 기준 = 차이 CI 하한 > 0 이고 NPS·DART 둘 다 / 하나만 / 없음 → '강함'/'부분'/'불충분').
시험: (1) NPS(원점 시점 상호 별칭 규칙)·DART·KIND 업종과의 5자리·3자리 일치, 기업 평균과 특허 가중 (2) 기업 규모·특허 수 구간별 (3) 그룹 안 5자리 선택력(같은 연계표 그룹이 맞은 특허 안에서) (4) FREE-IPC의 변형(평활 강도, 제목+IPC 결합 등) 중 개발 표본에서 가장 강한 것을 미리 정한 규칙으로 고른 뒤 평가 표본에 적용(과적합 방지). 
결론 문장: "제공 라벨은 공개 자료로 만들 수 있는 가장 강한 배정보다 기업 업종 일치에서 X(NPS)·Y(DART)". 응답 20줄.`, { label: '확인:근거A', phase: '확인·그림', effort: 'high' }),
  () => agent(`${COMMON}
[임무 — 그림 수정] ${OUT}\\figs\\ 에서 편집 메모 §C의 조치를 모두 반영해 다시 렌더하라(원본 PNG는 figs\\_v1\\ 로 옮겨 보관).
- 모든 그림에서 **제목 줄의 '그림 n' 번호를 빼라**(캡션은 문서에서 붙인다). 제목은 짧은 설명만.
- 그림 1(framework): 대조군 상자의 "공개 최강" 삭제, TXT-FREE 40.4%·FREE-IPC 48–54% 표기 확인.
- 그림 4(within_code_divergence): (b)(c)를 t5_cases.csv의 같은 창(2008–09→2016–17) 값으로 다시 그리고 소제목 수정.
- 그림 3(necessity): 주석 "제공 − 공개 최강" → "제공 − 해설서 단독(TXT-FREE)". 그리고 FREE-IPC(공개 최강) 막대를 추가할 수 있게 데이터 자리를 마련(값은 확인 시험 결과로 나중에 채움 — 지금은 R11 사후 값 NPS·DART를 '사후' 표시로).
- 전 그림을 이미지로 직접 열어 글자 겹침·잘림·한글 깨짐 확인.
응답: 수정한 그림 목록과 확인 결과.`, { label: '그림수정', phase: '확인·그림', effort: 'high' }),
])

phase('수정')
const v2 = await agent(`${COMMON}
[확인 시험 결과] ${conf || '(없음 — confirm_R12 폴더 확인)'}
[그림 수정 결과] ${figs || '(없음)'}
[임무 — 원고 v2] manuscript_v1_from_response.md 를 고쳐라:
1) 근거 A 문장 전부(요약·1.3·3.6.1·3.6.4·4.1·4.4 등)를 확인 시험 결과에 맞게 재서술 — '공개 최강' 대비 수치를 주로 쓰고, TXT-FREE 대비 수치는 보조. 결과가 약하면 근거 A의 강도 표기를 낮추고, "이 데이터가 아니면 안 되는 이유"의 구성(정의상 필수 과제·기업 업종 일치·T1·5개국)을 정직하게 재배열.
2) 그림 번호를 원고 순서대로 캡션으로 붙이고(그림 n. 설명 — 출처), 그림 파일명 매핑표를 편집 메모에 남겨라.
3) 편집 메모 §D의 남은 확인(부처 명칭은 '지식재산처' 기준, 출처 표의 대조표 발행기관 등)을 처리하고, 처리 못 한 것은 메모에 남겨라.
4) 20쪽 이내 유지(§E 계산 갱신).
응답 = 원고 v2 전문(마크다운, 제출본 부분 / 부록(숫자 출처표) / 편집 메모 순). 파일 ${OUT}\\manuscript_v2.txt 저장 시도.`, { label: '원고v2', phase: '수정', effort: 'high' })

return { conf, figs, v2 }

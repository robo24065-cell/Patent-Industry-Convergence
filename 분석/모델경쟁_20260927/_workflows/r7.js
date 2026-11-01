export const meta = {
  name: 'r7-contest-fit-essentiality',
  description: '7라운드: 공고문 부합성 감사 + 9개 트랙의 제공 특허-산업 융합데이터 필수성(제거·대체·5개국) 검증 + 제공 KSIC 라벨 외부 타당성 + 추가 데이터 확보 → 종합',
  phases: [
    { title: '검증', detail: '부합성 감사, 필수성 3그룹, 라벨 외부타당성, 데이터 확보' },
    { title: '종합', detail: 'R7 보고서와 보고서 구성 확정' },
  ],
}

const B = String.raw`C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`
const R = B + String.raw`\분석\모델경쟁_20260927`
const EXT = R + String.raw`\_shared\ext`

const COMMON = `
[배경] 「특허-산업 융합데이터 활용 아이디어 공모전」(지식재산처·KISTA), 제출 2026-11-13. 지금까지 9개 예측 트랙이 잠긴 최종시험을 통과했다(성적표: ${R}\\R6_결과_20261004.md).
[필독] **${R}\\_shared\\PROTOCOL_R7.md (이번 라운드 규칙 — 사용자 지시, 공고문 핵심, 필수성 검증 정의, 외부 타당성, 다운로드 허용)**, ${R}\\_shared\\README.md, PROTOCOL_R2.md §5(외부데이터 규칙), PROTOCOL_R6.md(공식 성적표·경고), ${R}\\R6_결과_20261004.md
[트랙 위치] T1 ${R}\\T1_산업대기실\\R3 (공식, c=2019), T3 ${R}\\T3_기술셀부상예측\\R3 (공개연도 공식) 및 R2, T4 ${R}\\T4_특허포기예측\\R2(공식)·R3, N4 ${R}\\N4_특허행태_기업폐업조기경보, N5 ${R}\\N5_표준특허선언레이더, N6 ${R}\\N6_정부RnD기업_첫특허전환, N7 ${R}\\N7_특허기업_정부RnD첫수행, N8 ${R}\\N8_특허분쟁표적레이더, N9 ${R}\\N9_정부RnD과제_특허성과예측. 각 폴더의 final_spec_*.json·final_run_*.py·FINAL_REPORT_*.md·experiments.csv를 읽어라.
[절대 규칙]
- 각 트랙의 동결 파일·결과 파일은 **읽기만**. 쓰기는 각 트랙 폴더의 ESSENTIAL_R7\\ 하위 또는 지정 폴더에만. 원본 DB·_shared 문서 수정 금지(sqlite mode=ro). 스크래치패드 금지. 수집파이프라인\\collect_kosis_new.py 수정 금지.
- 개발 원점이 주 평가. 최종시험 구간 재학습 비교는 '사후 분석'으로만 표기하고 공식 성적은 바꾸지 않는다. 최종시험 결과로 모형을 고치지 않는다.
- 제공 데이터 내용 외부 전송 금지, 키 값 출력 금지, 외부 호출은 간격·백오프. 자기가 띄운 프로세스만 종료. 메모리 4GB/프로세스, n_jobs≤2, GPU·pip 금지, 스크립트 1회 9분 이내(긴 재학습은 쪼개서).
- 숫자는 실제 실행 결과만. 한국어 기록.
`

const ESS = [
  { key: 'labelout', tracks: 'T1 산업 대기실, N6 R&D기업 첫 특허, N9 정부R&D 과제 특허 성과', note: '세 트랙 모두 결과변수가 제공 KSIC 라벨(50개 산업 특허)로 정의된다 → "과제 정의 자체가 제공 데이터 필요"를 명시하고, 피처 쪽 제거·대체 시험을 한다. T1은 학습 행이 수백만이라 음성 추출·원점 축소로 계산량을 줄이되 그 사실을 기록.' },
  { key: 'cell', tracks: 'T3 기술셀 부상(공개연도·출원연도), N5 표준특허 레이더', note: 'T3의 단위(KSIC×IPC 메인그룹 셀)와 해외 동일셀 선행 신호는 제공 5개국 KSIC 라벨 데이터에서 나온다. 대체 시험은 셀의 KSIC를 연계표로 배정해 셀을 다시 만들고(결과변수도 그 셀 기준으로 재정의해야 하면 두 버전 모두 보고), 5개국 제거 시험은 해외 모멘텀 피처 제거로. N5는 5개국 패밀리 폭 피처 제거.' },
  { key: 'registry', tracks: 'T4 특허 포기, N4 기업 폐업, N7 특허기업 R&D 진입, N8 특허 분쟁', note: '이 트랙들은 등록원부·국민연금·NTIS 비중이 크다. 제공 데이터의 역할(모집단 정의, KSIC 피처, 5개국 family, 제공 제목·IPC)을 정직하게 목록화하고 제거·대체 시험으로 기여를 잰다. 기여가 작으면 작다고 쓰고, "제공 데이터가 필수인 트랙"과 "보조로 쓰는 트랙"을 구분해 판정.' },
]

const ESS_SCHEMA = { type: 'object', additionalProperties: false, required: ['tracks'],
  properties: { tracks: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['id', 'roles', 'definitional', 'removal', 'substitution', 'kr_only', 'verdict', 'files'],
    properties: { id: { type: 'string' }, roles: { type: 'string', description: '제공 데이터가 쓰이는 곳(단위·모집단·결과·피처) 목록' },
      definitional: { type: 'string', description: '과제·단위·결과가 제공 데이터 없이 정의 불가능한가와 근거' },
      removal: { type: 'string', description: '제공 데이터 유래 피처 묶음 제거 시 AUC·skill 변화(개발 원점별, 사후 최종 분석 포함 시 표기)' },
      substitution: { type: 'string', description: '제공 KSIC 라벨을 연계표 배정으로 대체 시 변화와 연계표 일치율' },
      kr_only: { type: 'string', description: '해외 레코드 유래 피처 제거 시 변화(해당 없으면 해당 없음)' },
      verdict: { type: 'string', description: '필수 / 핵심 기여 / 보조 중 판정과 보고서 문장' },
      files: { type: 'string' } } } } } }

phase('검증')

const fitP = agent(`${COMMON}
[임무 — 공고문 부합성 감사] 공고문 PDF(${B}\\[붙임1] 특허-산업 융합데이터 활용 아이디어 공모전 공고문.pdf)를 직접 읽고, 9개 트랙과 지금까지의 산출물(R2~R6 결과 문서, _shared\\ext 데이터, _shared_report 표)이 요구사항을 얼마나 충족하는지 **항목별 대조표**를 만들어라: 주제(트렌드 분석 방법론·아이디어), 제공 데이터 필수 활용, 외부 데이터 결합(공고문 예시·붙임5 항목별 사용 여부), 분석 프레임워크 결과물 6종(데이터 연계 프레임워크·연계 결과·방법론·소스코드·분석 결과·시사점/정책 제언), 1차·2차 평가 기준 각 항목, 예시 1·2와의 차별성, 보고서 형식(hwp/doc 20쪽), 유의사항(출처 표기·개인정보·데이터 공유 금지 — 예: NTIS·RoS·KPSS 출처 표기, 기업명 노출 수준).
빠진 것·약한 것(예: '기업 간 거래정보' 미사용, KMAPSneo 미사용, 트렌드 '분석'보다 '예측' 위주인지)을 찾아 보완안을 제시. 산출: ${R}\\_shared_report\\공고문_부합성_대조표.md. 최종 응답 25줄.`,
  { label: '부합성', phase: '검증', effort: 'high' })

const essP = parallel(ESS.map((g) => () => agent(`${COMMON}
[임무 — 제공 특허-산업 융합데이터 필수성 검증 (PROTOCOL_R7 §2)]
대상: ${g.tracks}
참고: ${g.note}
각 트랙 폴더의 ESSENTIAL_R7\\ 에 스크립트·결과·essential_summary.md를 남겨라. 연계표(20131002 IPC-KSIC, KSIC 9차 기준일 수 있음 → 10차 대응 필요)는 읽기만. 더 새 공식 연계표가 공개 다운로드로 있으면 _shared\\ext\\r7_ 접두어로 받아 써도 된다(R7_SOURCES 기록 — 다른 에이전트와 파일명 충돌 피하기 위해 r7_${g.key}_ 접두어).
제거·대체·5개국 시험을 동결 챔피언 설정 그대로(피처만 바꿔) 개발 원점에서 재학습해 비교하라. 결과는 AUC와 skill 둘 다. 최종 응답은 스키마로.`,
  { label: `필수성:${g.key}`, phase: '검증', schema: ESS_SCHEMA, effort: 'high' })))

const validP = agent(`${COMMON}
[임무 — 제공 KSIC 라벨의 외부 타당성 (PROTOCOL_R7 §3)]
폴더 ${R}\\_shared_report\\label_validity\\ 를 만들어 여기에만 쓴다.
1) KR 특허(제공 데이터)의 KSIC 라벨 vs 공식 IPC-KSIC 연계표 배정(메인IPC 기준, KSIC 차수 맞춤)의 일치율을 KSIC별로 잰다(이전 측정: 3자리 41%, κ 0.36 — 재확인).
2) 기업 단위: 기업의 특허 KSIC 구성(제공 vs 연계표)이 기업의 등록 업종(국민연금 사업장 업종코드, DART induty_code, NTIS 수행기관 정보 등 원점 시점 값)과 얼마나 일치하는지.
3) 산업 단위: KSIC별 연간 특허 수(제공 라벨 vs 연계표 배정)와 KOSIS 광업제조업조사 5자리 출하·종사자·사업체 수, 무역(ksic_hs6_map 경유 수출입)의 상관·선행(개발 기간).
4) 결론 문장: "제공 데이터의 KSIC 라벨은 공식 연계표로 재현되지 않으며(일치율 X), 실제 산업 지표와의 정합도 Y만큼 높다/낮다" — 결과가 불리해도 그대로 써라.
산출: label_validity.md + csv. 최종 응답 20줄.`,
  { label: '라벨타당성', phase: '검증', effort: 'high' })

const dataP = agent(`${COMMON}
[임무 — 분석 시스템용 추가 데이터 확보 + IPO 레이더 파일럿]
1) 필요한 공개 데이터를 받는다(PROTOCOL_R7 §4): KRX KIND 신규상장·상장폐지 목록(가능한 기간 전체), 더 새 IPC-KSIC 공식 연계표(있으면), KSIC 10차 해설·차수 연계표, 공고문 붙임5의 무역통계·산업연관표 중 아직 없는 것. 받을 때마다 _shared\\ext\\R7_SOURCES.md 에 출처·수집일·범위·용도. 파일명은 r7_data_ 접두어. 로그인 필요 시 받지 말고 _shared\\ext\\USER_REQUESTS_R7.md 에 기록.
2) IPO 레이더 파일럿(R6 종합이 제안 — ${R}\\_ideas_R6\\IDEAS_R6.md 참고): 특허 보유 비상장 법인이 2년 안에 상장하는가를, 제공 KSIC 특허 포트폴리오 + 등록원부 + 국민연금(원점 시점 상호 별칭 규칙, PROTOCOL_R6 §2) + NTIS로 예측. 개발 원점에서만 파일럿, 최종시험 설계는 가장 최근 관측 가능한 구간으로 사전 고정안만 작성(채점 금지). **제공 데이터가 필수인지**(제거·대체 시험)도 파일럿에서 확인. 폴더 ${R}\\_ideas_R7\\ 에만 쓰고 IDEAS_R7.md 정리.
최종 응답 20줄.`,
  { label: '데이터확보', phase: '검증', effort: 'high' })

const [fit, ess, valid, data] = await Promise.all([fitP, essP, validP, dataP])

phase('종합')
const report = await agent(`${COMMON}
[7라운드 결과]
부합성 감사: ${fit}
필수성: ${JSON.stringify(ess.filter(Boolean), null, 2)}
라벨 외부 타당성: ${valid}
데이터 확보·IPO 파일럿: ${data}
[임무] 파일(_shared_report\\공고문_부합성_대조표.md, 각 트랙 ESSENTIAL_R7\\, _shared_report\\label_validity\\, _shared\\ext\\R7_SOURCES.md, _ideas_R7\\IDEAS_R7.md)을 직접 대조해 ${R}\\R7_결과_20261005.md 작성.
포함:
1) 공고문 부합성 판정과 보완 과제(우선순위).
2) **트랙별 제공 데이터 필수성 표**: 정의상 필수 여부 | 제거 시 하락 | 연계표 대체 시 하락 | KR만 쓸 때 하락 | 판정(필수/핵심/보조). 사용자 기준("이 데이터가 필수여야 한다")으로 **보고서 본문에 올릴 트랙**과 내릴 트랙을 다시 정하라.
3) 제공 KSIC 라벨의 외부 타당성 결론.
4) 보고서 핵심 메시지(3문장)와 '데이터 연계 프레임워크' 그림 설명 — 제공 데이터가 중심에 있고 외부 데이터가 어디에 붙는지.
5) 다음 검증 과제.
숫자는 파일 실측. 한국어. Write 후 최종 응답 30줄 이내.`,
  { label: '종합', phase: '종합', effort: 'high' })

return { fit, ess, valid, data, report }

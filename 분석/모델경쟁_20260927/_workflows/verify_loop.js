export const meta = {
  name: 'verify-loop',
  description: '검증 무한루프: 매 바퀴 계획 → 데이터 품질·편향 감사/결과 오류 감사/기존 트랙 재검증/새 아이디어·데이터 확보(최대 4개 병렬) → 반박 검증 → 원장 기록 (STOP.txt 또는 최대 바퀴까지)',
  phases: [
    { title: '계획', detail: '원장을 보고 이번 바퀴 대상 선정' },
    { title: '실행', detail: '감사·재검증·발굴 병렬' },
    { title: '반박', detail: '오류·새 결과를 반박 시도' },
    { title: '기록', detail: '원장·발견·정오표 갱신' },
  ],
}

const B = String.raw`C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`
const R = B + String.raw`\분석\모델경쟁_20260927`
const L = B + String.raw`\분석\검증루프_20261012`
const RPT = B + String.raw`\분석\보고서_20261012`
const MAX_CYCLES = 30

const COMMON = `
[배경] 「특허-산업 융합데이터 활용 아이디어 공모전」(지식재산처·KISTA), 제출 2026-11-13. 보고서 초안 v3(20쪽): 원고 ${RPT}\\manuscript_v2_from_response.md, 숫자 고정표 ${RPT}\\facts\\facts.csv(+facts_supplement.csv). 9개 예측 트랙(T1 산업대기실, T3 기술셀부상, T4 특허포기, N4 기업폐업, N5 표준특허, N6 R&D기업 첫특허, N7 특허기업 R&D진입, N8 특허분쟁, N9 정부R&D 과제성과)이 잠긴 최종시험을 통과. 전체 이력: ${R}\\R6_결과_20261004.md ~ R10_결과_20261005.md, 12라운드 확인시험 ${RPT}\\confirm_R12_summary.md, 11라운드 ${RPT}\\verify_R11_summary.md. 공용 규칙: ${R}\\_shared\\README.md, PROTOCOL_R2.md(§5 외부데이터)~PROTOCOL_R7.md.
[사용자 지시(그대로)] "계속 무한루프 검증해 줘. 예전에 하던 기존 아이디어 검증 및 새 아이디어 발굴도 계속. 필요한 데이터 있으면 찾아서 다운받아 쓰고. **다운받았다고 막 쓰지 말고, 그 데이터가 정확한지, 전체가 아니라 어느 한쪽으로 치우친 잘못된 데이터가 아닌지 확인하고, 결과에 오류가 있는지도 확인**하고 계속 무한루프로 돌려." 또 이전부터: 제공 **특허-산업 융합데이터가 필수**여야 하고 그 산업 연결의 장점을 활용, 공모전 주제(외부 데이터 결합 트렌드 분석) 부합.
[원장] ${L}\\LEDGER.jsonl(바퀴별 기록), FINDINGS.txt(확인된 발견), ERRATA.txt(원고 수정 제안), STOP.txt(있으면 중단).
[절대 규칙]
- 원자료·트랙 폴더·원고·facts는 **읽기만**. 쓰기는 이번 바퀴 폴더 안에서만(.md 쓰기가 막히면 .txt/.json/.csv). sqlite는 mode=ro.
- **공식 성적은 바꾸지 않는다.** 이미 채점한 최종시험 구간으로 모형을 고치지 않는다. 새 과제의 최종시험은 아직 아무도 보지 않은 최근 구간만, 사전 고정 후 1회.
- **새 데이터 검증 의무**: 내려받은 데이터는 쓰기 전에 (a) 공식 총량·공개 통계와 대조 (b) 연도·국가·산업·기업규모·지역별 분포와 결측·중복 (c) 연결(매칭) 시 선택 편향(연결된 쪽과 안 된 쪽 비교) (d) 시점 정합(스냅샷·사후 수정·발표 시차) 을 확인하고 결과를 남긴다. 치우침이 있으면 그 사실과 영향을 기록하고 보정하거나 쓰지 않는다.
- 내려받기는 로그인 없는 공개 데이터만(PROTOCOL_R2 §5.2, R7 §4). 회원가입 필요한 곳은 ${L}\\USER_REQUESTS.txt 에 적기만. 제공 데이터 내용 외부 전송 금지, API 키 값 출력 금지, 과속 금지.
- 판정 기준은 실행 전에 PREREG.txt에. 불리한 결과도 그대로. 메모리 4GB/프로세스, n_jobs≤2, GPU·pip 금지, 스크립트 9분 이내(긴 작업은 쪼개기), 자기 프로세스만 종료. 한국어.
`

const SEED_BACKLOG = `
[초기 점검 후보 — 원장에 이미 처리된 것은 건너뛰고, 필요하면 새로 추가]
데이터 품질·편향: 국민연금 사업장(3인 이상·2015-11~, 기업규모·업종별 연계 편향) / NTIS(수집 총액 16.3조 대 공식 약 19조 — 보안과제 누락, 성과 보고 지연·대학 편중) / 국세청 상태(2013 이전 좌측 절단, 폐업일 타당성) / KIPRIS 인용 800건 표본 편향 / 제공 데이터 CN 추세가 WIPO 공식과 불일치(순위상관 0.13)·JP 2021 결손 → 5개국 지표·RTA 영향 / 사업자 권리목록(2026 스냅샷, 법인만, 개인 RC:001 제외) / DART·KIND(상장·외감 대기업 편향) / KOSIS 10차·11차 단절(2020) / Comtrade HS 개정 단절·상대국 범위 / IPC 2022–23 신설코드 소급 / 제공 데이터 중복·결측·연도 이상 / 등록원부 납부 완결성·소멸 판정 / R8 fam_link.parquet 결함(공개번호 오독)을 쓴 다른 분석.
결과 오류: 원고 v3의 모든 숫자·문장을 결과 파일과 대조(장별로 나눠) / facts_mismatch 15건 처리 / 공식 9개 트랙 최종 결과 독립 재현(아직 안 한 N6·N7·N8·N9 우선) / 그림 CSV와 원고 숫자 일치.
기존 트랙 재검증: T3(CN 품질 보정·KR≤1건 셀 지배), T1(미래 등록 조건부 모집단·이름 매칭 레이블), N8(양성 108건·자기 제기 권리범위확인심판 혼입), N9(보고 지연판 선택), N4(국민연금 키 보정·국세청 절단), 산업 집계 지표의 소수 대기업 지배.
새 아이디어: 제공 데이터의 산업 연결(KSIC 5자리 × 5개국 × 기업)이 없으면 성립하지 않고, 아직 잠글 최근 결과 구간이 남은 과제(이전 후보: _ideas_R5·R6·R7·R8의 보류안 포함). 반드시 필수성(공개 배정 대체) 시험과 데이터 편향 점검 포함.
`

const PLAN_SCHEMA = { type: 'object', additionalProperties: false, required: ['stop', 'reason', 'tasks'],
  properties: { stop: { type: 'boolean' }, reason: { type: 'string' },
    tasks: { type: 'array', maxItems: 4, items: { type: 'object', additionalProperties: false,
      required: ['id', 'kind', 'target', 'why', 'instructions'],
      properties: { id: { type: 'string', description: '예: C03-T1' }, kind: { type: 'string', enum: ['data_audit', 'result_audit', 'track_reverify', 'new_idea', 'data_acquire'] },
        target: { type: 'string' }, why: { type: 'string' }, instructions: { type: 'string', description: '실행 에이전트가 바로 일할 수 있는 구체 지시(입력 파일, 확인 항목, 판정 기준, 산출물)' } } } } } }
const EXEC_SCHEMA = { type: 'object', additionalProperties: false, required: ['task_id', 'status', 'summary', 'findings', 'files'],
  properties: { task_id: { type: 'string' }, status: { type: 'string', enum: ['done', 'partial', 'blocked'] }, summary: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['claim', 'severity', 'evidence', 'affects', 'proposed_fix'],
      properties: { claim: { type: 'string' }, severity: { type: 'string', enum: ['error', 'warning', 'ok', 'new_result'] },
        evidence: { type: 'string', description: '수치·파일 경로' }, affects: { type: 'string', description: '영향받는 원고 절·트랙·데이터' }, proposed_fix: { type: 'string' } } } },
    files: { type: 'string' } } }
const VER_SCHEMA = { type: 'object', additionalProperties: false, required: ['confirmed', 'notes'],
  properties: { confirmed: { type: 'boolean', description: '반박 실패 시 true. 불확실하면 false' }, notes: { type: 'string' } } }

const RESUME_C02 = `
[재개 안내] 이 작업은 직전 시도에서 사용 한도로 중간에 끊겼다. 폴더에 이미 있는 PREREG.txt(판정 기준 — 고치지 말 것, 바꿔야 하면 '변경 기록'에 사유와 함께 추가만)와 중간 산출물(csv·json·scripts·cache·download)을 먼저 읽고 검토한 뒤 이어서 끝까지 진행하라. 남은 중간 파일은 그대로 믿지 말고 생성 스크립트·입력과 맞는지 확인하고 쓴다.`
const MS_NOTE = `
[원고 최신본] 메인 세션이 ERRATA를 반영해 원고를 고칠 수 있다. ${RPT}\\manuscript_v3.md 가 있으면 그것을 현재 원고로 보고 감사하라(줄 번호도 그 파일 기준). 없으면 manuscript_v2_from_response.md. ERRATA.txt에 '반영됨' 표시가 있는 항목은 다시 제안하지 말 것.`
const history = []
for (let c = 1; c <= MAX_CYCLES; c++) {
  const cid = `C${String(c).padStart(2, '0')}`
  const dir = `${L}\\${cid}`
  phase('계획')
  const plan = await agent(`${COMMON}
${SEED_BACKLOG}
[직전 바퀴 요약] ${history.length ? history.slice(-5).join('\n---\n') : '(첫 바퀴)'}
${c >= 3 ? MS_NOTE + '\n' : ''}[임무 — ${cid} 계획] ${L}\\STOP.txt 가 있으면 stop=true. 아니면 LEDGER.jsonl·FINDINGS.txt·ERRATA.txt를 읽고, 아직 안 한 것 중 **보고서 신뢰성에 영향이 큰 순서로** 이번 바퀴 작업 3~4개를 고른다(가능하면 종류를 섞을 것: 데이터 감사 1 + 결과 감사 1 + 재검증 또는 새 아이디어 1 + 필요 시 데이터 확보). 같은 대상을 반복하지 말되, 직전 바퀴에서 '부분/막힘'이었던 것은 이어서 할 수 있다. 작업 폴더는 ${dir}\\<작업id>\\. 더 할 가치 있는 일이 정말 없으면 stop=true와 이유.`,
    { label: `${cid}:계획`, phase: '계획', schema: PLAN_SCHEMA, effort: 'high' })
  if (!plan || plan.stop) { log(`${cid}: 중단 — ${plan ? plan.reason : '계획 실패'}`); break }
  log(`${cid}: ${plan.tasks.map((t) => `${t.kind}:${t.target}`).join(' | ')}`)

  phase('실행')
  const results = await pipeline(
    plan.tasks,
    (t) => agent(`${COMMON}
[이번 작업] ${t.id} (${t.kind}) — ${t.target}
[이유] ${t.why}
[지시] ${t.instructions}
[폴더] ${dir}\\${t.id}\\ (만들어서 여기에만 쓴다)${c === 2 ? RESUME_C02 : ''}${c >= 3 ? MS_NOTE : ''}
데이터 확보·새 데이터 사용 시 [새 데이터 검증 의무] (a)~(d)를 반드시 수행해 결과를 data_check.txt에 남겨라. 결과 감사는 원천 파일에서 직접 재계산해 대조하라. 발견은 severity로 구분(error=원고·공식 결과에 틀린 것, warning=조건·한계 누락, ok=확인됨, new_result=새 결과). 최종 응답은 스키마로.`,
      { label: `${t.id}:실행`, phase: '실행', schema: EXEC_SCHEMA, effort: 'high' }),
    async (res, t) => {
      if (!res) return { t, res: null, ver: [] }
      const toVerify = res.findings.filter((f) => f.severity === 'error' || f.severity === 'new_result')
      const ver = await parallel(toVerify.map((f, k) => () => agent(`${COMMON}
[반박 검증] 아래 발견을 **반박하는 것이 목표**다. 원천 파일을 직접 열어 다시 계산하라. 불확실하면 confirmed=false.
작업: ${t.id} ${t.target}
발견: ${JSON.stringify(f)}
작업 폴더(읽기): ${dir}\\${t.id}\\ — 쓰기는 ${dir}\\${t.id}\\verify_${k}\\ 에만.`,
        { label: `${t.id}:반박${k}`, phase: '반박', schema: VER_SCHEMA, effort: 'high' })))
      return { t, res, ver: toVerify.map((f, k) => ({ f, v: ver[k] })) }
    },
  )

  phase('기록')
  const summary = await agent(`${COMMON}
[${cid} 결과]
계획: ${JSON.stringify(plan)}
실행·반박: ${JSON.stringify(results.filter(Boolean).map((r) => ({ id: r.t.id, kind: r.t.kind, target: r.t.target, status: r.res && r.res.status, summary: r.res && r.res.summary, findings: r.res && r.res.findings, verify: r.ver })), null, 1)}
[임무 — 원장 기록] ${L} 의 파일에 **추가(append)** 하라(기존 내용 보존):
1) LEDGER.jsonl 에 한 줄 JSON: {"cycle":"${cid}","tasks":[...id·kind·target·status],"confirmed_errors":n,"warnings":n,"new_results":n,"notes":"..."}
2) FINDINGS.txt 에 반박을 통과한(confirmed) error·new_result와 주요 warning을 "[${cid}][작업id][severity] 내용 — 근거 파일" 형식으로.
3) ERRATA.txt 에 원고에 반영할 수정 제안(절, 현재 문장/숫자 → 수정안, 근거)을. 반박에 실패한(confirmed=false) 오류 주장은 넣지 말 것.
최종 응답: 이번 바퀴 요약 10줄(다음 계획자가 읽는다).`,
    { label: `${cid}:기록`, phase: '기록', effort: 'high' })
  history.push(`${cid}: ${summary || '(기록 실패)'}`)
  log(`${cid} 완료`)
}
return { cycles: history.length, last: history.slice(-3) }

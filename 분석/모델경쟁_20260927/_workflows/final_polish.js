export const meta = {
  name: 'final-polish',
  description: '최종안 다듬기: 원고 4구간 병렬 편집(숫자·단서·형식 보존) → 구간별 반박 검증·수선 → 조립·빌드(20쪽·배치 점검) → 독립 최종 감사',
  phases: [
    { title: '편집', detail: '구간별 가독성 다듬기' },
    { title: '검증', detail: '숫자·단서·주장 강도 대조, 필요 시 원문 복원' },
    { title: '조립', detail: '합치기·docx/PDF 빌드·20쪽·배치' },
    { title: '감사', detail: '독립 최종 감사' },
  ],
}

const B = String.raw`C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012`
const P = B + String.raw`\drafts\polish`
const CTX = `
[배경] 「특허-산업 융합데이터 활용 아이디어 공모전」(지식재산처·KISTA) 분석 결과 보고서, 20쪽 이내(hwp 또는 doc), 제출 2026-11-13. 심사 배점: 창의성(데이터 융합·연결의 독창성) 40, 발전 가능성 40 등. 원고는 20회 넘는 검증 바퀴(분석\\검증루프_20261012 의 FINDINGS.txt·ERRATA.txt)로 숫자·단서를 맞춘 상태라, 내용은 정확하지만 단서 괄호가 쌓여 읽기 무겁다. 이번 일은 '읽기 좋은 최종안'으로 다듬는 것이다.
[파일] 작업 폴더 ${P} : base.md(동결 원고, sha1 앞 8자리 base_sha1.txt), S1_orig.md~S4_orig.md(구간 원문), appendix_tail.md(부록, 손대지 않음), numcheck.py(대조기: python numcheck.py S1 → RESULT PASS/FAIL).
[절대 규칙]
1. 숫자(값·CI·%·건수·연도·절 번호 포함)를 바꾸거나 빼거나 새로 넣지 않는다. 모든 숫자 토큰이 원문과 같은 횟수로 남아야 한다(numcheck.py PASS 필수).
2. 주장을 강화하지 않는다. 단서(경계·조건부·공식 외·참고용·사후·연관이지 인과 아님·0 포함·보수적·가설·민감·하한 등)는 짧게 줄일 수는 있어도 빼지 않는다. 판정어(같다·부분·작음·조건부·견고·영향 없음 등)는 그대로 둔다.
3. 새 사실·새 해석·새 인용·새 기업명을 넣지 않는다. 문장을 다른 구간으로 옮기지 않는다(구간 안 순서 조정은 가능).
4. 형식 유지: 제목 줄(#, ##, ###)은 한 글자도 바꾸지 않는다. 표의 행·열 수, 그림 자리표시 '[그림 n: 파일]', 캡션 번호('**표 n.**', '그림 n.'), 글머리표·번호 목록 구조, 굵게(**…**) 표지어, 첫 '---' 구분선은 유지한다. 표 셀 문구는 다듬어도 되지만 숫자는 그대로. 참고문헌 항목과 '데이터 출처' 표(이용 조건 표기)는 손대지 않는다. 표지의 '팀명: ○○○'은 그대로 둔다.
5. 파일은 UTF-8, 줄바꿈 LF로 쓴다.
[다듬기 방향] 심사위원(비전공자 포함)이 빠르게 읽고 '제공 데이터가 왜 필요한지, 무엇을 예측했고 얼마나 맞았는지, 어디에 쓰는지'를 바로 잡도록 한다.
- 한 문장에 한 생각. 긴 괄호(대략 40자 넘는 것)는 문장으로 풀거나 꼭 필요한 단서만 남긴다. 같은 말 반복을 없앤다. 명사 나열보다 동사 문장.
- 각 절 첫 문장에서 '무엇을 했고 결론이 무엇인지'를 먼저 말한다(이미 그렇다면 둔다).
- 용어 통일: skill, AUC, 기준선, 공식 성적, 최종시험, 개발 원점, 공개 최강 배정, 제공 라벨.
- 분량: 글자 수를 원문보다 3–10% 줄인다(늘리면 안 된다 — 20쪽이 빠듯하다).
- 문체: 보고서 평서문(-다). 영어 약어 풀이는 원문에 있는 대로.
`
const SECTIONS = [
  { id: 'S1', desc: '표지·요약·1장(문제 정의, 공고문 예시와 차별성, 제공 데이터의 본질, 연계 프레임워크)', extra: '요약은 심사위원이 가장 먼저 읽는다. 문제 → 데이터 융합 방식 → 핵심 결과 → 왜 이 데이터인가 → 시사점 흐름이 한눈에 들어오게 하라(숫자는 모두 유지).' },
  { id: 'S2', desc: '2장(전진 검증·잠긴 최종시험, 지표, 의심 장치, 필수성 시험 설계, 전처리)과 3장 서두·3.1 트렌드 분석', extra: '2.5 전처리 글머리는 단서가 가장 많이 쌓인 곳이다. 단서를 빼지 말고 문장 구조로 정리하라.' },
  { id: 'S3', desc: '3.2 T3 기술셀 부상, 3.3 T1 산업 대기실, 3.4 N9·N6 정부 R&D, 3.5 보조 트랙 박스', extra: '트랙마다 정의 → 공식 성적(AUC·skill·상위 k) → 단서 → 미래 전망 순서가 보이게 하라.' },
  { id: 'S4', desc: '3.6 제공 데이터 필수성(정의·근거 A·B·C·대체 시험 종합), 4장(시사점·활용 시나리오·발전 방향·한계·결론), 데이터 출처, 참고문헌', extra: '데이터 출처 표와 참고문헌은 손대지 않는다. 4.4 한계는 항목 번호와 단서를 모두 유지하되 문장을 짧게 하라.' },
]

const EDIT_SCHEMA = { type: 'object', additionalProperties: false, required: ['section', 'numcheck', 'chars_before', 'chars_after', 'notes'],
  properties: { section: { type: 'string' }, numcheck: { type: 'string', enum: ['PASS', 'FAIL'] }, chars_before: { type: 'number' }, chars_after: { type: 'number' }, notes: { type: 'string' } } }
const VER_SCHEMA = { type: 'object', additionalProperties: false, required: ['section', 'verdict', 'numcheck_final', 'problems', 'notes'],
  properties: { section: { type: 'string' }, verdict: { type: 'string', enum: ['PASS', 'FIXED', 'FAIL'] }, numcheck_final: { type: 'string', enum: ['PASS', 'FAIL'] },
    problems: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } } }
const ASM_SCHEMA = { type: 'object', additionalProperties: false, required: ['pages', 'last_page_bottom_pt', 'docx', 'pdf', 'layout_issues', 'fixes', 'notes'],
  properties: { pages: { type: 'number' }, last_page_bottom_pt: { type: 'number' }, docx: { type: 'string' }, pdf: { type: 'string' },
    layout_issues: { type: 'array', items: { type: 'string' } }, fixes: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } } }
const AUD_SCHEMA = { type: 'object', additionalProperties: false, required: ['verdict', 'issues', 'notes'],
  properties: { verdict: { type: 'string', enum: ['ready', 'fix_needed'] },
    issues: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['severity', 'where', 'problem', 'suggestion'],
      properties: { severity: { type: 'string', enum: ['error', 'warning', 'style'] }, where: { type: 'string' }, problem: { type: 'string' }, suggestion: { type: 'string' } } } },
    notes: { type: 'string' } } }

phase('편집')
const results = await pipeline(
  SECTIONS,
  (s) => agent(`${CTX}
[임무 — ${s.id} 편집] 구간: ${s.desc}. ${s.extra}
${P}\\${s.id}_orig.md 를 읽고, 위 규칙대로 다듬은 결과를 ${P}\\${s.id}_edit.md 로 쓴다. 쓴 뒤 그 폴더에서 'python numcheck.py ${s.id}' 를 실행해 PASS가 될 때까지 고친다(FAIL 원인이 숫자면 원문 표현을 되살린다). 최종 응답: 구간 id, numcheck 결과, 글자 수 전후, 주요 변경 3–6줄.`,
    { label: `${s.id}:편집`, phase: '편집', schema: EDIT_SCHEMA, effort: 'high' }),
  (ed, s) => agent(`${CTX}
[임무 — ${s.id} 반박 검증] 편집자가 ${P}\\${s.id}_edit.md 를 만들었다(편집자 보고: ${JSON.stringify(ed)}). 원문 ${P}\\${s.id}_orig.md 와 문장 단위로 대조해 규칙 위반을 찾는 것이 목표다. 'python numcheck.py ${s.id}' 를 직접 실행하고, 그다음 사람이 읽듯 비교하라: (a) 단서·판정어가 빠지거나 약해졌는가 (b) 주장이 강해졌는가(예: '같다'→'우위', '경계' 삭제, 조건 생략) (c) 의미가 바뀐 문장(대상·기간·비교 기준이 바뀜) (d) 새 사실·기업명 (e) 형식 깨짐(표·캡션·자리표시) (f) 오히려 읽기 나빠진 곳.
문제가 있으면 ${s.id}_edit.md 에서 그 문장만 원문 의미로 고치고(필요하면 원문 문장을 그대로 되살림) numcheck PASS를 다시 확인한다. 최종 응답: verdict(PASS=문제 없음, FIXED=고쳐서 통과, FAIL=통과 못함), 최종 numcheck, 발견·수정한 문제 목록.`,
    { label: `${s.id}:검증`, phase: '검증', schema: VER_SCHEMA, effort: 'high' }).then((v) => ({ s, ed, v })),
)
const ok = results.filter(Boolean)
log(`편집·검증: ${ok.map((r) => `${r.s.id}=${r.v ? r.v.verdict + '/' + r.v.numcheck_final : 'null'}`).join(', ')}`)

phase('조립')
const asm = await agent(`${CTX}
[임무 — 조립·빌드·배치 점검] 구간 검증 결과: ${JSON.stringify(ok.map((r) => ({ id: r.s.id, verdict: r.v && r.v.verdict, numcheck: r.v && r.v.numcheck_final })))}.
1) ${P} 에서 S1_edit.md + S2_edit.md + S3_edit.md + S4_edit.md + appendix_tail.md 를 순서대로 이어 붙여 manuscript_polished.md 를 만든다. 검증이 FAIL이거나 _edit 파일이 없는 구간은 _orig 를 대신 쓴다(어느 구간을 대신했는지 보고).
2) 빌드: Bash에서 cd "${B}\\build"; export NODE_PATH="$(npm root -g)"; FIG_NATIVE=1 node md2docx.js "${P}\\manuscript_polished.md" ../figs "${P}\\final_candidate.docx" 를 실행하고, PowerShell에서 powershell -NoProfile -File "${B}\\build\\render.ps1" "${P}\\final_candidate.docx" 로 PDF와 쪽수를 얻는다(Word COM, 기존 Word 창을 닫지 말 것; 자기 Word 프로세스만 정리).
3) PyMuPDF(fitz)로 쪽마다 본문 마지막 블록 하단 y(쪽 바닥 790pt 위 블록만)를 재고, 쪽 이미지(70dpi)를 ${P}\\pages\\ 에 저장해 직접 보며 배치 문제를 찾는다: 20쪽 초과, 큰 빈 공간(그림·표가 다음 쪽으로 밀림), 제목만 쪽 끝에 남음, 표가 어색하게 잘림, 캡션이 그림과 떨어짐.
4) 21쪽 이상이면: 늘어난 구간의 문장을 더 줄이거나(규칙 준수, numcheck PASS 유지) 그 구간의 일부 문단을 원문으로 되돌려 20쪽을 맞춘다. 20쪽이면 큰 빈 공간을 줄이는 작은 문장 조정만 한다. 고친 _edit 파일마다 numcheck 를 다시 돌려 PASS를 확인하고 다시 조립·빌드한다.
최종 응답: 쪽수, 마지막 쪽 하단 pt, docx·pdf 경로, 남은 배치 문제, 적용한 수정.`,
  { label: '조립·빌드', phase: '조립', schema: ASM_SCHEMA, effort: 'high' })

phase('감사')
const aud = await agent(`${CTX}
[임무 — 독립 최종 감사] 조립 결과: ${JSON.stringify(asm)}. 너는 편집에 참여하지 않았다. ${P}\\manuscript_polished.md(최종 후보)와 ${P}\\base.md(동결 원고)를 비교하고, ${P}\\final_candidate.pdf 를 읽어 다음을 판정한다.
(1) 제출 요건: 20쪽 이내, 표지·요약·장 구성(① 분석 프레임워크 ② 분석 결과 ③ 시사점)이 공고문 요구와 맞는가, 쪽 번호·그림·표 번호가 이어지는가.
(2) 사실 보존: 본문(부록 앞) 숫자 토큰이 원고와 같은가(직접 스크립트로 대조), 요약·4.1 결론·표 11의 핵심 주장이 원고와 같은 강도인가, 단서(N5 경계, T1 사후 조건부, 근거 B 이원 군집 0 포함, 공식 외 전향 확인 등)가 살아 있는가.
(3) 읽기 품질: 심사위원 관점에서 여전히 막히는 문장, 용어 불일치, 문단 흐름 단절. 고칠 문장을 구체적으로 제안하라(숫자 불변 조건에서).
파일은 고치지 말고(읽기만), 문제를 severity(error=사실·요건 위반, warning=단서 약화·배치, style=문장)로 보고한다. 최종 응답: verdict(ready 또는 fix_needed), issues, notes.`,
  { label: '최종 감사', phase: '감사', schema: AUD_SCHEMA, effort: 'high' })

return { sections: ok.map((r) => ({ id: r.s.id, edit: r.ed, verify: r.v })), assembly: asm, audit: aud }
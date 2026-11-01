export const meta = {
  name: 'model-competition-r6',
  description: '6라운드: 신규 N8(특허 분쟁 표적 레이더)·N9(정부R&D 과제 특허 성과 예측) 구축·최종시험 + N7 국민연금 연결키 영향·위약 재검증 + 7라운드 아이디어 → 종합',
  phases: [
    { title: '구축·검증·발굴', detail: 'N8·N9 구축, N7 재검증, 아이디어 파일럿' },
    { title: '최종시험', detail: 'N8 t0=2022, N9 t=2022 — 1회' },
    { title: '종합', detail: 'R6 보고서' },
  ],
}

const B = String.raw`C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`
const R = B + String.raw`\분석\모델경쟁_20260927`
const EXT = R + String.raw`\_shared\ext`

const COMMON = `
[배경] 「특허-산업 융합데이터 활용 아이디어 공모전」(지식재산처·KISTA). 제공 데이터(KSIC 세세분류 50개별 5개국 특허, 각 특허에 KSIC 라벨)와 참가자 수집 외부 데이터 결합이 주제. 제출 2026-11-13.
[사용자 지시] "계속 테스트·체킹해 줘." 사용자는 정확도를 0.67·0.88 같은 직관적 수치로 보고 싶어 한다 → **AUC와 상위 k 적중률을 skill과 함께 반드시 보고**. 단, 쉬운 과제로 숫자만 부풀리지 말고 강한 기준선 대비 skill도 보고.
[필독] ${R}\\_shared\\README.md, PROTOCOL_R2.md(§5), PROTOCOL_R3.md, PROTOCOL_R4.md, PROTOCOL_R5.md, **PROTOCOL_R6.md(공식 성적표·국민연금 연결키 규칙·N8/N9 규칙)**, ${R}\\R5_결과_20261004.md, ${R}\\_ideas_R5\\IDEAS_R5.md
[절대 규칙]
- 자기 폴더에만 쓴다. 원본 DB·_shared 문서·다른 트랙 파일 수정 금지(sqlite mode=ro). 스크래치패드 금지. 수집파이프라인\\collect_kosis_new.py 수정 금지.
- 🔒 최종시험 구간 결과변수 계산·조회 금지. 결과 구간이 F 이후와 겹치는 원점은 개발·학습에 쓰지 않는다.
- 피처는 원점 시점 정보만. 사후 목록(2026 CORP_APPLICANT 상호, 2026 권리목록 등)을 키·피처로 쓸 때 편향을 점검하고 기록. NTIS·OpenAlex·국세청은 현재 스냅샷(준-정합).
- 제공 데이터 내용을 외부로 보내지 않는다. 키는 API키_기록.txt에서 읽고 값 출력 금지. 외부 호출은 간격·백오프.
- 자기가 띄운 프로세스만 종료. 메모리 4GB/프로세스, n_jobs≤2, GPU·pip 금지, 스크립트 1회 9분 이내.
- 숫자는 실제 실행 결과만. 한국어 기록.
[재시도 안내 — 2026-10-05] 직전 시도가 사용량 한도로 중간에 끊겼다. 네 폴더에 이전 시도가 남긴 파일(PLAN.md, experiments.csv, 캐시, 스크립트, final_run_r6.py 등)이 있을 수 있다. 먼저 확인하고 **이어서** 하라(처음부터 다시 만들지 말 것). 남은 파일이 규칙을 지켰는지(특히 최종시험 구간 미조회) 확인하고, 이미 끝난 실험은 다시 돌리지 말고 기록을 이어 쓴다. --mode final이 이미 실행된 흔적(final_result_r6.json 등)이 있으면 다시 실행하지 말고 그 결과를 감사·보고하라.
`

const NEW = [
  { id: 'N8', dir: 'N8_특허분쟁표적레이더', name: '특허 분쟁 표적 레이더 (J2)',
    spec: `IDEAS_R5.md J2 설계 그대로: 단위 = t0 말에 살아 있는 등록특허·실용신안(제공범위), 결과 = 무효심판·권리범위확인심판 예고등록 ∈ (t0, t0+2](등록원부 right 사건). 기준선 M0와 묶음(M1 등록원부 이력, M2 외부 특허 단위 신호 RoS·KPSS·ETSI·NTIS)은 IDEAS_R5 정의. 파일럿 XGB skill 0.423(9/9), AUC 0.84–0.90, 분쟁의 28–40% 상위 1%에서 포착. 파일럿 코드 ${R}\\_ideas_R5\\ (봉인 캐시 _ideas_R4\\cache는 날짜<2023-01-01).
주의: 양성이 매우 희소(0.4–1.0‰) — 권리자 군집 CI, 후속 분쟁 하위집단(사전 등록), 강한 기준선(M0+자기·권리자 과거 분쟁) 대비 skill. 외부 신호는 XGB에선 롤백·로지스틱에선 채택이었으니 모형족별로 정직하게.
🔒 최종: t0=2022 → 예고등록 2023–24 (F=2023), 학습 t0 2008–2020.` },
  { id: 'N9', dir: 'N9_정부RnD과제_특허성과예측', name: '정부R&D 과제의 50개 산업 특허 성과 조기 예측 (J1)',
    spec: `IDEAS_R5.md J1 설계 그대로: 단위 = NTIS 과제(과제연도 단위), 결과 = 그 과제의 성과로 등록된 제공범위 KR 특허(출원 t..t+1). 기준선 M0와 묶음(과제 텍스트→KSIC50 정의문 대응 점수, 제공 50산업 성과 이력 등)은 IDEAS_R5 정의. 파일럿 XGB skill 0.278(7/7), AUC 0.88–0.90, 상위 1% 적중 74–83%. **보고 지연판을 주 모형**(IDEAS_R5 권고, 사후 결정임을 공개), 원판 병기.
주의: NTIS 성과 보고 지연으로 최근 성과가 과소 관측될 수 있다 — 출원연도별 연결률로 보고 완결도를 추정해 기록. 수행기관 군집 CI. 강한 기준선(M0+제공 50산업 성과 이력) 대비 skill. N6(기업 단위 첫 특허)와 단위가 다름을 명시.
🔒 최종: 과제연도 t=2022 → 출원 2022–23 (F=2022). 학습은 결과창이 2021 이하로 끝나는 과제연도만.` },
]

const IMP_SCHEMA = { type: 'object', additionalProperties: false,
  required: ['champion_start', 'champion_end', 'n_experiments', 'accepted', 'rolled_back', 'external_results', 'frozen', 'notes'],
  properties: { champion_start: { type: 'string' }, champion_end: { type: 'string' }, n_experiments: { type: 'number' }, accepted: { type: 'string' },
    rolled_back: { type: 'string' }, external_results: { type: 'string' }, frozen: { type: 'string' }, notes: { type: 'string' } } }
const FIN_SCHEMA = { type: 'object', additionalProperties: false,
  required: ['ran', 'audit', 'metric', 'model_auc', 'baseline_auc', 'skill', 'ci_low', 'ci_high', 'strong_baseline_skill', 'dev_skill', 'topk', 'placebo', 'plain_accuracy', 'external_in_model', 'verdict'],
  properties: { ran: { type: 'boolean' }, audit: { type: 'string' }, metric: { type: 'string' }, model_auc: { type: 'number' }, baseline_auc: { type: 'number' },
    skill: { type: 'number' }, ci_low: { type: 'number' }, ci_high: { type: 'number' }, strong_baseline_skill: { type: 'string' }, dev_skill: { type: 'number' },
    topk: { type: 'string' }, placebo: { type: 'string' }, plain_accuracy: { type: 'string' }, external_in_model: { type: 'string' }, verdict: { type: 'string' } } }

phase('구축·검증·발굴')

const newRuns = pipeline(
  NEW,
  (s) => agent(`${COMMON}
[너의 트랙] ${s.id} ${s.name} (6라운드 신규)
[폴더] ${R}\\${s.dir}\\ (만들어서 여기에만 쓴다)
[설계] ${s.spec}
[임무] 1) 파일럿 코드를 출발점으로 DATA_NOTES.md 확정, 파일럿 M0·챔피언을 개발 원점에서 재현(experiments.csv). 2) 묶음 사전 선언(PLAN.md) 후 **최소 10개 실험**(채택/롤백). 3) 동결: final_spec_r6.json + final_run_r6.py (--mode dev / final(🔒, 결과 구간 < F 레이블로만 학습, AUC·skill·강한 기준선 skill·상위 k·군집 부트스트랩 95% CI를 final_result_r6.json에) / future). **final·future 실행 금지**, dev만. 4) R6_NOTES.md. 최종 응답은 스키마로.`,
    { label: `구축:${s.id}`, phase: '구축·검증·발굴', schema: IMP_SCHEMA, effort: 'high' }),
  (imp, s) => agent(`${COMMON}
[너의 트랙] ${s.id} ${s.name}
[폴더] ${R}\\${s.dir}\\
[구축 결과] ${JSON.stringify(imp || { note: '구축 결과 없음 — 폴더 직접 확인' })}
[임무 — 감사 후 🔒최종시험 1회] 실행 전 감사: (a) 레이블 결과 구간 < F (b) 피처 시점(등록원부 사건일 ≤ t0, NTIS 레코드 연도·연구비 이듬해 규칙, RoS·KPSS·ETSI 날짜 조건, 사후 목록 키) (c) 기준선 M0 고정 (d) CI 군집. 위반 시 최소 수정 → --mode dev 재확인 → 기록. 성능 보고 모형 수정 금지. --mode final 1회(실행 오류만 수정), 독립 재계산 대조, **위약(층 내 레이블 섞기) 200회 이상**, 강한 기준선 대비 skill, 사전 등록 하위집단, --mode future, FINAL_REPORT_R6.md. 최종 응답은 스키마로.`,
    { label: `최종시험:${s.id}`, phase: '최종시험', schema: FIN_SCHEMA, effort: 'high' }).then((f) => ({ s, imp, f })),
)

const n7P = agent(`${COMMON}
[임무 — N7 재검증 (PROTOCOL_R6 §1·§2)] 대상: ${R}\\N7_특허기업_정부RnD첫수행\\ (동결 모형·최종 결과는 읽기만). 쓰기는 N7_특허기업_정부RnD첫수행\\VERIFY_R6\\ 에만.
1) 국민연금 연결키 영향: N7은 N4와 같은 키(사업자번호 앞 6자리 + 2026-07 상호)를 쓴다. N4 재검증 방식(N4_특허행태_기업폐업조기경보\\VERIFY_R5\\ — 사업장 정체성으로 옛 상호 별칭 연결)으로 모든 원점 패널을 다시 만들고 동결 스펙으로 재학습해 최종시험 skill이 얼마나 바뀌는지 추정(공식 성적은 바꾸지 않음, 영향만 보고). 'NPS 미가입 기업 AUC 0.806'이 오분류 구간에서 나온 것인지 확인.
2) 위약 200회 이상(층 내·완전 무작위), 3) 강한 기준선·하위집단·보정, 4) 누수 재탐색.
VERIFY_R6\\robustness_table.md와 요약을 남겨라. 최종 응답: 보정 후 skill·위약·경고 15줄.`,
  { label: '검증:N7', phase: '구축·검증·발굴', effort: 'high' })

const ideaP = agent(`${COMMON}
[임무 — 7라운드 아이디어 발굴] 1~6라운드 교훈을 반영해 기존 트랙(T1·T3·T4·N4~N9)과 겹치지 않고, 아직 잠글 최근 결과 구간이 남은 새 과제 2~3개를 구상하고 개발 원점에서 파일럿을 실제로 돌려라. 공모전 적합성(외부데이터 결합·KSIC 다리·트렌드 분석·정책 활용)과 정확도(AUC·상위 k)·강한 기준선 대비 skill을 함께 평가. 폴더 ${R}\\_ideas_R6\\, IDEAS_R6.md에 정리. 최종시험 구간 채점 금지. 최종 응답 20줄.`,
  { label: '발굴', phase: '구축·검증·발굴', effort: 'high' })

const [runs, n7, ideas] = await Promise.all([newRuns, n7P, ideaP])
runs.filter(Boolean).forEach((r) => log(`${r.s.id}: ${r.f ? `최종 AUC ${r.f.model_auc} (기준선 ${r.f.baseline_auc}), skill ${r.f.skill} [${r.f.ci_low}, ${r.f.ci_high}]` : '결과 없음'}`))

phase('종합')
const report = await agent(`${COMMON}
[6라운드 결과]
신규: ${JSON.stringify(runs.filter(Boolean).map((r) => ({ id: r.s.id, build: r.imp, final: r.f })), null, 2)}
N7 재검증: ${n7}
아이디어: ${ideas}
[임무] 파일(N8·N9 final_result_r6.json·FINAL_REPORT_R6.md, N7\\VERIFY_R6\\, ${R}\\_ideas_R6\\IDEAS_R6.md)을 직접 대조해 ${R}\\R6_결과_20261004.md 작성.
포함: 1) N8·N9 최종시험(AUC·skill·강한 기준선 skill·상위 k·위약) 2) N7 재검증 결과와 경고 3) **전체 공식 성적표**(9개 트랙: AUC·기준선 AUC·skill·CI·상위 k·재검증 경고·출처) 4) 보고서 구성 권고 갱신 — 본문 주축/보조, 대표 정확도 수치(AUC·상위 k)로 무엇을 내세울지 5) 7라운드 후보.
숫자는 파일 실측. 한국어. Write 후 최종 응답 30줄 이내.`,
  { label: '종합', phase: '종합', effort: 'high' })

return { runs: runs.filter(Boolean).map((r) => ({ id: r.s.id, final: r.f })), n7, report }

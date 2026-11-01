export const meta = {
  name: 'r13-figures-print-size',
  description: '13라운드: 보고서 그림을 최종 인쇄 크기(본문 폭 17cm, 글자 7pt 이상)로 다시 그리기 + 지형도·프레임워크 재설계 → 20쪽 유지 확인',
  phases: [{ title: '그림', detail: '도식 2·데이터 그림 6 재렌더' }, { title: '조판확인', detail: 'docx 재조판·쪽수·가독성 확인' }],
}

const B = String.raw`C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`
const OUT = B + String.raw`\분석\보고서_20261012`

const COMMON = `
[배경] 공모전 보고서(A4 20쪽 이내) 원고 v2: ${OUT}\\manuscript_v2_from_response.md. 현재 docx(${OUT}\\보고서_초안_v2_20261012.docx, PDF 같은 이름)는 그림 높이를 8.7cm로 줄여 20쪽에 맞췄는데, **그림 1(fig01_framework)과 그림 4(fig05_rta_map_5x50)의 글자가 인쇄 시 읽기 어렵다.** 그림 폴더 ${OUT}\\figs\\ (make_data_figs.py, make_diagrams.py, _style.py, 각 CSV). 원고에서 쓰는 그림은 [그림 n: 파일명] 자리표시로 확인.
[목표] 모든 그림을 **최종 인쇄 크기 그대로** 다시 그린다: figsize를 실제 cm로(본문 폭 17cm = 6.69in; 반폭 그림은 쓰지 않음), dpi=300, 그림 안 최소 글자 7pt(축 눈금·주석 포함), 제목은 그림 안에 넣지 말 것(캡션은 문서에서). 높이는 그림마다 정하되 **전체 그림 높이 합이 지금(그림 8개 × 8.7cm ≈ 70cm)을 넘지 않게** — 20쪽을 지키기 위해서다. 가로로 긴 레이아웃을 우선.
- 그림 1(프레임워크): 17cm × 약 10cm 가로형으로 재배치, 상자 글자 7–8pt, 설명 문장 줄이기(세부 수치는 표 3에 있음).
- 그림 4(5개국×50산업 RTA): 50행 히트맵은 인쇄 크기에서 읽을 수 없다 → 재설계. 예: 한국 RTA 상위 15개·하위 10개 산업만 5개국 점 그래프/작은 히트맵, 또는 50개 산업을 두 열(25행씩)로 접은 가로형 17cm × 11cm 이하. 원고 3.1.1에 나오는 숫자(25/50 RTA>1, 1위 7개, 화장품·OLED·무인기 등)가 그림에서 보이게.
- 나머지 그림도 같은 규칙으로 다시 렌더(값은 바꾸지 말 것). 원본은 figs\\_v2\\ 로 옮겨 보관.
[절대 규칙] 숫자·데이터 변경 금지, 원자료는 읽기만, 지정 폴더에만 쓴다. 한글 글꼴 Malgun Gothic, 유니코드 마이너스 대신 하이픈. 한국어.
`

phase('그림')
const figs = await agent(`${COMMON}
[임무] 위 목표대로 원고에서 쓰는 그림 전부를 다시 렌더하고, 각 PNG를 이미지로 직접 열어 7pt 이상·겹침 없음·잘림 없음을 확인하라. 그림별 최종 물리 크기(cm)를 표로 응답하라.`, { label: '그림재렌더', phase: '그림', effort: 'high' })

phase('조판확인')
const check = await agent(`${COMMON}
[그림 재렌더 결과] ${figs}
[임무] 변환기 ${OUT}\\build\\md2docx.js 는 환경변수 FIG_NATIVE=1이면 그림을 그린 물리 크기(300dpi 기준) 그대로 배치한다. 다음을 실행해 확인하라(bash):
  cd "${OUT}\\build"; export NODE_PATH="$(npm root -g)"; FIG_NATIVE=1 node md2docx.js ../manuscript_v2_from_response.md ../figs ../보고서_초안_v3.docx
  powershell -NoProfile -ExecutionPolicy Bypass -File render.ps1 "<보고서_초안_v3.docx 절대경로>"   (MS Word로 PDF와 쪽수)
그다음 PyMuPDF(fitz)로 그림이 있는 쪽을 90dpi로 렌더해 이미지로 직접 보고 가독성을 확인하라. **20쪽을 넘으면** 그림 높이를 줄이거나(가로형 유지) 원고 편집 메모 §E의 축소 순서를 따르되, 원고 텍스트를 고쳐야 하면 ${OUT}\\manuscript_v3.md 로 새로 저장(막히면 응답에 변경 내용을 명확히)하라. 응답: 최종 쪽수, 그림별 배치 크기, 가독성 확인 결과, 원고 변경 사항.`, { label: '조판확인', phase: '조판확인', effort: 'high' })

return { figs, check }

# 분석 코드 안내서 (CODE_GUIDE)

이 저장소(`robo24065-cell/Patent-Industry-Convergence`)의 분석 코드가 어디에 무엇이 있고, 무엇을 읽어 무엇을 쓰는지 정리한 문서다.

- 근거: 각 스크립트의 머리말(docstring·주석), 경로 상수, 파일 입출력 호출을 읽어 정리했다. 코드에 직접 적혀 있지 않은 해석은 **(추정)** 으로 표시했다. 확인하지 못한 점은 맨 끝 [9절](#9-추정확인-못-한-점)에 모았다.
- 이 문서를 만들 때 코드를 실행하지 않았다. 데이터·결과표·키는 저장소에 없다(README.md, 공모전 서약서 6항).
- 기준: `main` 커밋 `5c94f0a`(추적 파일 2,392개).
- 제출 전 점검은 [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md), 패키지 생성은 `tools/make_code_package.py`.

## 목차

0. [표기](#0-표기)
1. [한눈에 보기](#1-한눈에-보기)
2. [실행 전제(공통)](#2-실행-전제공통)
3. [모델경쟁: 트랙별 구조](#3-분석모델경쟁_20260927--트랙별-구조)
4. [_shared: 공용 메타 빌드와 외부데이터](#4-_shared--공용-메타-빌드와-외부데이터)
5. [_workflows: 오케스트레이션 스크립트](#5-_workflows--오케스트레이션-스크립트)
6. [검증루프: 바퀴·작업 구조와 명명 규칙](#6-분석검증루프_20261012--바퀴작업-구조와-명명-규칙)
7. [보고서: 빌드·그림·숫자 고정표](#7-분석보고서_20261012--빌드그림숫자-고정표)
8. [초기 단계 코드](#8-초기-단계-코드)
9. [추정·확인 못 한 점](#9-추정확인-못-한-점)

---

## 0. 표기

| 표기 | 뜻 |
|---|---|
| `<ROOT>` | 팀 PC 프로젝트 루트 `C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`. 코드 1,026개 파일이 이 경로를 절대경로로 쓴다 |
| `<MC>` | `<ROOT>\분석\모델경쟁_20260927` |
| `<META>` | `<MC>\_shared\meta\*.parquet` — 제공 융합데이터 5개국 메타(`_shared/build_meta.py` 산출) |
| `<EXT>` | `<MC>\_shared\ext\` — 외부데이터 수집·가공 산출물 폴더 |
| 🔒 | 코드 안의 '봉인' 표시. 최종시험 구간의 결과를 읽지 못하게 막는 가드 |
| 1R, R2 … R13 | 1라운드, 2라운드 … 13라운드 |
| 진입 스크립트 | 그 단계를 시작하거나 대표하는 실행 파일. 화살표(→)는 실행 순서 |

## 1. 한눈에 보기

### 1.1 폴더와 파일 수

| 경로 | 파일 | 내용 |
|---|---:|---|
| `분석/모델경쟁_20260927/` | 918 | 예측 트랙 T1–T6·N1–N9, 아이디어 파일럿 `_ideas_R2`–`_ideas_R8`, 공용 `_shared`, 공동 분석 `_shared_report`, 워크플로 `_workflows` |
| `분석/검증루프_20261012/` | 1,358 | 검증 루프 바퀴 C01–C20 |
| `분석/보고서_20261012/` | 7 | 보고서 빌드 `build/`, 숫자 고정표 `facts/`, 그림 `figs/` |
| `분석/*.py`, `분석/대기실_파일럿/`, `분석/fusion/` | 95 + 7 + 2 | 초기 단계(PADI 탐색, 수집 점검, T1 전신 파일럿) |
| 루트 | 5 | `README.md`, `.gitignore`, `.gitattributes`, `.githooks/guard.py`, `.githooks/pre-commit` |

확장자별: `.py` 2,354 · `.js` 22 · `.ps1` 12. 트랙별 파일 수는 3절 제목에 적었다. 이 안내서와 함께 추가된 `docs/`, `tools/`, `requirements.txt`는 위 숫자에 넣지 않았다.

### 1.2 작업 흐름(시간 순)

1. **초기 탐색**(2026-08, `분석/*.py`): PADI 지표, 수집 데이터 점검, 적대적 검증. 날짜는 머리말의 '적대 감사(2026-08-11)', '사전 등록(2026-08-12)' 표기에서 읽었다.
2. **1라운드**(`모델경쟁_20260927/T1`–`T6` 최상위): 1단계 탐색(`v*`, `exp*`) → 2단계 감사·동결(`s2*`, `audit/`, `final_spec.json`) → 3단계 홀드아웃 평가(`final_run.py --mode holdout`, `stage3_*`, `verify_holdout.py`).
3. **2–6라운드**: T1·T3·T4는 `R2/`·`R3/`·`R4/`로 이어진다. 새 트랙은 N1–N3(R2), N4–N5(R4), N6–N7(R5), N8–N9(R6)에 생겼다. 라운드마다 아이디어 파일럿 `_ideas_R<n>/`가 있다. 5라운드는 레드팀 재검증(`VERIFY_R5/`)이다.
4. **7–13라운드**: 제공 데이터 필수성 검증(`ESSENTIAL_R7/`, `NONCIRC_R8/`, `TXT_R9/`, `_shared_report/`)과 보고서 작성(R11 초안, R12 수정, R13 그림 인쇄 크기). R6–R13은 `_workflows/r<n>.js`가 에이전트 작업을 묶는다.
5. **검증 루프**(`검증루프_20261012/C01`–`C20`, `_workflows/verify_loop*.js`): 데이터·결과 감사와 반박 검증.
6. **최종 다듬기·빌드**(`_workflows/final_polish.js`, `보고서_20261012/build/`).

### 1.3 공식 9개 트랙과 공식 성적 실행기

보고서 그림 2(`figs/make_data_figs.py`의 `fig02`)에 나오는 9개 트랙이다. '결과 파일'은 숫자 고정표(`facts/build_facts.py`)가 읽는 파일이다(저장소에는 없다).

| 단계 | 트랙 | 폴더(`<MC>` 아래) | 공식 최종시험 실행기 | 결과 파일 |
|---|---|---|---|---|
| 뜬다 | T3 기술셀 부상 | `T3_기술셀부상예측` | `R3/final_run_r3.py --mode final` | `R3/final_result_r3.json` |
| 들어온다 | T1 산업 대기실 | `T1_산업대기실` | `R3/final_run_r3.py --mode final` | `R3/final_result_r3.json` |
| 다툰다 | N8 특허 분쟁 표적 | `N8_특허분쟁표적레이더` | `final_run_r6.py --mode final` | `final_result_r6.json` |
| 버려진다 | T4 특허 포기 | `T4_특허포기예측` | `R2/final_run_r2.py --mode final` | `R2/final_result.json` |
| 쓰러진다 | N4 기업 폐업 | `N4_특허행태_기업폐업조기경보` | `final_run_r4.py --mode final` | `final_result_r4.json` |
| 표준이 된다 | N5 표준특허 선언 | `N5_표준특허선언레이더` | `final_run_r4.py --mode final --confirm-final` | `final_result_r4.json` |
| 특허↔R&D | N9 R&D 과제 특허 성과 | `N9_정부RnD과제_특허성과예측` | `final_run_r6.py --mode final` | `final_result_r6.json` |
| 특허↔R&D | N6 R&D기업 첫 특허 | `N6_정부RnD기업_첫특허전환` | `final_run_r5.py --mode final` | `final_result_r5.json` |
| 특허↔R&D | N7 특허기업 R&D 첫 수행 | `N7_특허기업_정부RnD첫수행` | `final_run_r5.py --mode final` | `final_result_r5.json` |

- T4는 3라운드에 새 최종시험이 없고(`R3/final_run_r3.py`에 `--mode final` 없음) 공식 성적은 R2 최종시험을 유지한다고 머리말에 적혀 있다.
- T2·T5·T6(1R만)과 N1–N3(R2)은 공식 성적표에 없다(탈락·보류로 추정).
- ⚠ `--mode final`은 '사전 고정 후 1회' 규칙으로 이미 채점이 끝난 단계다. 여러 실행기가 결과 파일이 있으면 재실행을 거부한다(N1·N2·N3·N8 등 머리말). 다시 돌리지 않는다.

## 2. 실행 전제(공통)

### 2.1 저장소에 없는 입력(팀 PC 로컬)

| 입력 | 위치 | 만드는 곳 / 주로 읽는 곳 |
|---|---|---|
| 제공 융합데이터 원본(50개 탭 구분 파일, 29.5GB) | `<ROOT>\공모전데이터\융합데이터_*\*.csv` | `_shared/build_meta.py`만 직접 읽는다 |
| 5개국 메타 parquet | `<META>` | `build_meta.py` 산출. T1–T6, N1·N3·N8·N9, `_ideas_*`, `_shared/ext`, `_shared_report`, 검증 루프가 읽는다 |
| 작업데이터 | `<ROOT>\공모전데이터\작업데이터\` — `kr_patents.parquet`, `kr_registered.parquet`, `kosis_광업제조업_50개산업.csv`, `kosis_광업제조업_2008_2024.csv` | 만드는 코드는 저장소에 없다(추정: 수집 단계 산출). T1·T6·N1·N2·N4–N9 |
| 공모전 폴더의 외부데이터 | `<ROOT>\공모전데이터\외부데이터\` — `DART\dart_company.sqlite`, `DART\corp_applicant.parquet`, `Corporate_20260720\CORP_APPLICANT.txt`, `KSIC해설\ksic50_definitions.json`, `20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx`(공식 IPC-KSIC 연계표), `산업분류별\*.csv`, `경기지표\*.csv`, `재무\*.csv`, `제3축\*.csv` | 대부분의 트랙(T1–T6, N1–N5, N7, N9)과 7라운드 이후 필수성 시험 |
| 수집 DB | `<ROOT>\수집파이프라인\registry_full.sqlite`(KR 등록원부), `business_rights.sqlite`(법인 권리목록, 2026 스냅샷) | 모두 `mode=ro` 읽기 전용으로 연다. 수집 코드 `collect_registry.py`·`collect_kosis_new.py`는 키가 들어 있어 저장소 밖에 있다(guard가 파일 이름으로 차단) |
| 연계표 | `<ROOT>\분석\fusion\ksic_hs6_map.csv`, `hs6_import_yearly.csv` | T1 R2, N2, N8, `_shared/ext/trade_*` |
| 외부데이터 수집본 | `<EXT>` | `_shared/ext/*.py`가 만든다(4절). 거의 모든 트랙 |
| API 키 | `<ROOT>\API키_기록.txt` | `_shared/ext`의 `*_common.py`·`*_util.py`·`*_keys.py`, `r3kosis_lib.py`(키를 `수집파이프라인\collect_kosis_new.py`에서 정규식으로 읽음) 등이 읽기만 한다. 값은 출력·저장하지 않는다 |
| 규칙·결과 문서 | `PROTOCOL_R*.md`, `SPLITS.md`, `PLAN.md`, `PREREG.txt`, `R*_결과_*.md`, `robustness_table.md` | 코드 주석이 참조. `.md`·`.txt`는 저장소에서 제외 |
| 동결 명세·중간 산출물 | `final_spec*.json`, `champion*.json`, `experiments.csv`, `cache*/`, `runs*/`, `preds/` | `.json`·`.csv`·`.parquet` 등은 제외. 그래서 동결 모형 설정은 저장소만으로는 알 수 없다 |

### 2.2 코드에서 반복되는 실행 관례

- **모드**: 동결 실행기는 대개 `--mode dev`(개발 원점 재현) · `--mode final`(🔒 최종시험 1회) · `--mode future`(최종시험 이후 전망). 1라운드는 `dryrun` · `holdout` · `future`이고, T3는 `selftest`(가짜 원점으로 코드 경로 점검), N3는 `prereg`(사전등록 점수)가 더 있다.
- **9분 제한과 종료코드 3**: 긴 단계는 캐시하면서 진행하고 시간 예산을 넘기면 종료코드 3으로 끝난다. 같은 명령을 다시 실행하면 캐시에서 이어 간다(T1 `final_run.py` 머리말 등). 같은 '스크립트 9분 이내' 규칙이 워크플로 프롬프트에도 있다(`_workflows/verify_loop*.js`). 에이전트 실행 도구의 명령 시간 제한 때문으로 보인다(추정).
- **자원**: 프로세스당 4GB 이하, `n_jobs ≤ 2`. 큰 단계는 하위 프로세스로 나눈다(예: T4 `final_run.py --step`, T5 `--stage build|fit`).
- **봉인 가드**: 최종시험 자료는 환경변수로만 열린다. 예: T3 `T3_HOLDOUT_UNLOCK`, `T3R2_FINAL_UNLOCK`, `T3R3_FINAL_UNLOCK`, T4 `T4_STAGE3_HOLDOUT`, T1 `T1_STRICT=1`·`T1_VIS=rgst`. 실행기가 해당 모드에서 스스로 켠다. 손으로 켜지 않는다.
- **동결 명세와 해시**: 모형·피처·하이퍼파라미터는 `final_spec*.json`에서 읽는다. N5(`lib_sha256`)·N6·N9(`code_sha256`) 실행기는 공용 코드 파일의 SHA-256을 명세와 대조한다. 동결 코드를 고치면 이 대조가 깨진다.
- **작업 폴더**: T3 1R의 `build_base.py`는 `../_shared/meta/*.parquet` 상대경로를 쓰므로 트랙 폴더에서 실행한다(`final_run.py` 머리말 '트랙 폴더에서'). T1은 `python R2/final_run_r2.py`처럼 트랙 폴더 기준 사용법이 적혀 있다. `분석/대기실_파일럿/*.py`는 `공모전데이터/...` 상대경로라 `<ROOT>`에서 실행한다(추정). 나머지는 대부분 `__file__` 기준 경로라 어디서 실행해도 된다.
- **한글 출력**: 대부분 표준출력을 UTF-8로 바꾼다. Windows 콘솔에서는 `python -X utf8`이 안전하다(`facts/build_facts.py` 머리말). 격리 모드 `-I`는 스크립트와 같은 폴더의 모듈 import를 막으므로, `figs/*.py`처럼 이웃 모듈(`_style.py`)을 쓰는 스크립트에는 쓰지 않는다.

### 2.3 Python·Node·외부 프로그램

| 항목 | 내용 |
|---|---|
| Python | 팀 PC 3.13.2(2026-11-02 확인). 제3자 패키지는 루트 [`requirements.txt`](../requirements.txt)(버전 미고정, 팀 PC 설치판을 주석으로 기록) |
| Node.js | 팀 PC v24.15.0. `보고서_20261012/build/md2docx.js`가 쓰는 npm 패키지는 **`docx` 하나**다(`fs`·`path`는 내장). 팀 PC는 전역 설치판 `docx@9.7.1`을 쓰고, 실행할 때 `NODE_PATH`를 전역 폴더로 준다: bash `export NODE_PATH="$(npm root -g)"`, PowerShell `$env:NODE_PATH = (npm root -g)`. 새 PC에서는 `npm install -g docx`(팀 PC와 같은 주 버전 9) 또는 `build/` 폴더에서 `npm install docx` |
| `_workflows/*.js` | npm 패키지를 쓰지 않는다. node로 직접 실행하는 파일이 아니다(5절) |
| `검증루프_*/…/build_test/md2docx.js` | 보고서 빌더 사본. 같은 `docx` 패키지를 쓴다 |
| MS Word | `render.ps1`(docx→PDF), `save_doc.ps1`(docx→.doc)이 COM 자동화로 쓴다 |
| 글꼴 | 맑은 고딕(`Malgun Gothic`) — 그림(`figs/_style.py`)과 docx 본문 |
| PyMuPDF(`fitz`) | PDF 쪽 이미지·배치 확인(워크플로 r13·final_polish, 검증 루프) |

### 2.4 저장소 관리 메모

- 커밋 가드: `.githooks/guard.py`(pre-commit). 데이터·압축 확장자, 5MB 초과, 허용 폴더(`보고서_20261012/figs/`, `facts/`) 밖 csv/json, 차단 파일 이름, 키 문자열(저장소 루트 `API키_기록.txt`가 있을 때)·하드코딩 키 패턴을 막는다. 처음 클론하면 `git config core.hooksPath .githooks`.
- `.gitignore`는 '전부 제외 후 `분석/` 아래 `.py`·`.js`·`.ps1`만 허용' 방식이다. 그래서 `docs/`, `tools/`, `requirements.txt`는 처음 추가할 때 `git add -f`가 필요했다. 이미 추적 중인 파일의 수정은 평소처럼 커밋된다. 이 폴더들에 새 파일을 더 넣을 때도 `-f`가 필요하다(또는 팀이 `.gitignore`에 예외를 추가).
- `.gitattributes`: `.py`·`.js`·`.md`·`.txt`는 LF, `.ps1`은 CRLF로 저장·체크아웃한다.

## 3. `분석/모델경쟁_20260927/` — 트랙별 구조

### 3.0 하위 폴더 이름 규칙

| 이름 | 뜻 |
|---|---|
| 트랙 최상위 | 1라운드. 1단계 탐색(`v<번호>_*.py`, `exp*.py`), 2단계 감사·개선(`s2*.py`, `audit/`, `make_spec.py`, `final_spec.json`), 3단계 홀드아웃(`final_run.py --mode holdout`, `stage3_*.py`, `verify_holdout.py`) |
| `R2/` `R3/` `R4/` | 2·3·4라운드. 챔피언/도전자 실험(실험 등록부 `*exps*.py`·`r<n>exp.py`, 실행기 `r<n>run.py`) → 동결(`make_spec*.py`) → 동결 실행기(`final_run_r<n>.py`) → 독립 재계산(`verify_final*.py`) |
| `VERIFY_R5/`(N7은 `VERIFY_R6/`, N6은 `audit_r5/`) | 5(6)라운드 레드팀 재검증: 캐시 없이 재현, 위약(레이블 섞기), 강화 기준선, 대형 출원인 제외, 누수 재탐색. 동결 파일은 읽기만 한다 |
| `ESSENTIAL_R7/` | 7라운드 필수성 시험: 제공 KSIC 라벨을 공식 IPC-KSIC 연계표(2013) 배정으로 바꾸거나 5개국 정보를 빼고 동결 챔피언을 같은 설정으로 다시 학습 |
| `NONCIRC_R8/` | 8라운드 '순환 없는' 필수성 시험: 결과를 제공 라벨로 정의하지 않는 과제로 비교 |
| `TXT_R9/` | 9라운드 두 번째 대체안: KSIC 해설서 텍스트로 만든 5자리 배정(TXT) |
| `audit/`, `probe/`, `s2/` | 감사, 탐침(결측·연계율 점검), 2단계 보조 모듈 |
| `cache*/`, `runs*/`, `preds/`, `specs/`, `res/`, `results/` | 산출물 폴더. 저장소에는 그 안의 `.py`만 있다. 예: T3 `ESSENTIAL_R7/cache_*/build_panel.py`는 T3 동결 빌더 `build_panel.py`와 바이트 단위로 같은 사본이다 |
| `*_archive_*` | 이전 판 보관(T3 `R3/B1_archive_E40/`, `R3/D_archive_preaudit/`) |

공통 파일 이름: `*lib.py`·`common.py`(경로·로더·평가 공용), `*_build*.py`·`build_*.py`(패널·캐시), `*_exp*.py`·`*run*.py`(실험), `*_robust.py`·`*_posthoc.py`·`*_info.py`(사후 점검, 채택 판단에 쓰지 않음), `*_audit*.py`(최종시험 전 감사).

### 3.1 T1 산업 대기실 (116개) — '법인 × KSIC 첫 진입' 예측

평가 단위는 (법인 × KSIC) 위험집합 쌍, 결과는 진입창 안에 그 법인이 그 KSIC로 처음 출원(등록·공개)하는지다.

| 라운드(폴더) | 진입 스크립트 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| 1R 준비(최상위) | `prep1.py` → `prep2.py` → `prep_app.py` (+`prep_ep.py`, `prep_insP.py`, `prep_sim.py`) | 원천 캐시: 범위안 특허·법인 목록·범위밖 자사출원·공개출원·해외(EP) 선행 | `registry_full.sqlite`, `business_rights.sqlite`, `dart_company.sqlite`, `작업데이터/kr_registered.parquet`·`kr_patents.parquet`, `<META>` | `cache/ins_pat`, `ins_applicant`, `firms`, `rl_pat`, `oos`, `ins_firm`, `firm_meta`, `app_pub`, `ep_firm` `.parquet` |
| 1R 실험 | `v0_baseline.py`…`v23_strict.py`, `exp.py`, `vstrict.py`, `s2.py`·`s2_cv.py`, `diag1–4.py`, `boot.py`, `clf_study.py` (공용 `common.py`, `lib.py`, `feats.py`, `data8.py`, `s2lib.py`) | 버전별 피처·모형 실험, 진단 | 위 캐시 | `cache/v*_scores.parquet`, `leaderboard.csv`, `bootstrap_ci.json` |
| 1R 동결·평가 | `final_model.py`, `final_run.py --mode dryrun\|holdout\|future`(`final_run_frozen_stage2.py`는 2단계 동결본), `stage3_verify.py`, `stage3_val_xgbbase.py`, `stage3_wrap.py`, `stage3_future_names.py`, `trend.py`·`trend_naive.py` | v31 동결 모형 실행, 독립 대조, 대기실 지수 초안 | `final_spec.json`, 위 캐시 | `final_cache/<mode>/`, `dryrun_result.json`, `holdout_result.json`, `eval_scores.parquet`, `holdout_verify.json`, `future_forecast.csv`, `future_candidates_top50.csv` |
| R2 | `R2/build_origin.py --c <원점>` → `R2/r2run.py --exp E.. --origin ..` / `--summarize` → `R2/make_spec.py E..` → `R2/final_run_r2.py --mode dev\|final\|future` → `R2/verify_final_r2.py`, `R2/final_plain_metrics.py` (공용 `r2.py`, 실험 등록부 `r2exps.py`, 외부 결합 `r2ext.py`) | 5년 진입창, 개발 원점 2008–2012, 최종시험 c=2017 | 1R 캐시, 산업연관표·무역·KOSIS(`r2ext.py`) | `R2/cache*/`, `R2/runs/`, `experiments.csv`, `final_spec_r2.json`, `dev_result.json`, `final_result.json`, `final_verify.json`, `future_forecast_r2.csv` |
| R3 (공식) | 준비 `R3/build_ipcx.py`, `R3/build_ext_src.py`, `R3/clf_ext.py --c ..`, `R3/build_r3.py --c ..` → 실험 `R3/r3run.py --exp R3E.. --origin ..` → `R3/r3nts.py`(국세청 폐업 반영) → `R3/make_spec_r3.py` → `R3/audit_r3.py` → `R3/final_run_r3.py --mode dev\|final\|future` → `R3/verify_final_r3.py` (공용 `r3.py`, `r3exps.py`, `r3ext.py`·`r3ext2.py`·`r3ext3.py`, `r3aux.py`). 순서는 머리말 기준(추정) | 4년 진입창, 개발 원점 2008–2012(+보조 2015), 최종시험 c=2019 → 2020–23 진입 | 1R·R2 캐시, `<META>`, `<EXT>`(OpenAlex·RoS·KOSIS 품목·국민연금·DART·국세청), 디자인·실용신안(`business_rights.sqlite`) | `R3/cache*/`, `R3/runs/FROZEN_*/`, `final_spec_r3.json`, `dev_result_r3.json`, `nps_layer_r3.json`, `final_result_r3.json`, `final_verify_r3.json`, `future_forecast_r3.csv`(그림 8) |
| R4 | `R4/t1_ntis_feats.py`, `R4/t1_ntis_feats2.py` → `R4/t1_layer.py` | R3 동결 예측 위에 NTIS(국가R&D) 2단 보정층 챔피언/도전자 | `R3/runs/FROZEN_dev/`, `<EXT>/ntis_*` | `R4/cache/ntis_o<c>.parquet`, `ntis_name_o<c>.parquet`, 실험 json, `experiments.csv` |
| VERIFY_R5 | `repro_t1.py`, `sens_t1.py`, `leak_t1.py`, `trunc_t1.py`, `devgap_t1.py`, `ci_sb_t1.py` | 레드팀: 원천에서 레이블 재구성, 민감도, 누수, 절단, 개발-최종 격차, 강화 기준선 CI | 동결 예측·캐시, sqlite(ro) | `VERIFY_R5/*.json`, `sens_t1_by_ksic.csv` |
| ESSENTIAL_R7 | `r7_t1_alt.py`, `r7_t1_ipcsub.py`, `r7_t1_defn.py` → `r7_t1_run.py` | 연계표 대체 필수성 시험. `R3/final_run_r3.py`의 `train_origin()`을 import해 그대로 학습 | R3 캐시, 연계표 xlsx | `ESSENTIAL_R7/runs/<변형>/`, `results_dev.csv`, `defn_t1.json` |
| NONCIRC_R8 | `a_build.py` → `a_eval.py` → `a_pool.py`; `b_build.py` → `b_eval.py`; `c_build.py` → `c_eval.py`; `summarize.py` (공용 `nc_common.py`) | (A) 라벨 무관 결과: 법인의 새 IPC 서브클래스 진입, (B) 외부 결과: 국민연금 등록 업종 변화, (C) 대칭 2×2 교차평가 | `<META>`, N4 `cache/app_firm.parquet`, `_shared_report/label_validity/cache_kr_cw.parquet`, 국민연금(`<EXT>`) | `NONCIRC_R8/cache/`, `res/A_*.json`, `res/A_summary.csv`, `res/C_summary.csv`, `res/C_delta.csv` |
| TXT_R9 | `t0_txt.py`, `t0b_txtfree.py` → `b_feat.py` → `c_eval.py` → `c_pool.py` → `summarize.py`; 기업 업종 일치 `s1_firm.py 2019\|2015` (공용 `r9_common.py`) | TXT 배정 대체안 시험, 기업 등록업종(NPS·DART·KIND) 일치 | NONCIRC_R8 캐시, KSIC 해설서 json, `_ideas_R7/cache` | `TXT_R9/cache/kr_txt*.parquet`, `res/C_*.json`, `res/C_summary.csv`, `res/C_pool_summary.csv`, `res/s1_*`, `res/t0*_agree.json` |

### 3.2 T3 기술셀 부상 (127개) — 'KSIC × IPC 메인그룹 셀'의 KR 출원 급성장 예측

셀별로 원점 t0 뒤 3년 KR 출원 성장이 상위 10%인지(부상)를 맞힌다. 머리말 기준 1R은 출원연도, R3 공식판은 공개연도 기준이다.

| 라운드(폴더) | 진입 스크립트 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| 1R(최상위) | 빌드 `build_base.py` → `build_panel.py` → `build_feat2.py`(2단계 보조 `build_feat3.py`) (공용 `t3lib.py`, `harness.py`, `models.py`, `s2lib.py`) · 실험 `v*_*.py`, `s2_v*.py`, `candidate_v22.py`, `abl1.py`, `leadlag.py` · 감사 `audit_dup.py`, `audit_ties.py`, `audit_trunc.py`, `diag1–4.py` · 동결 `final_run.py --mode dryrun\|holdout\|selftest\|future` · 대조 `verify_holdout.py` (`forecast_future.py`, `holdout_eval.py`는 '대체됨' 표시) | v30 동결 모형(로지스틱 + XGB + 음이항 꼬리확률의 순위평균). `final_run.py`가 빌드 3단계를 하위 프로세스로 부른다 | `<META>`(상대경로 `../_shared/meta`, 트랙 폴더에서 실행), duckdb | `cache*/base_rows.parquet`, `panel_ccy.parquet`, `feat2.parquet`, `feat_<t0>.parquet`, `dryrun_result.json`, `holdout_result.json`, `verify_holdout.json`, `future_forecast.csv`, `future_forecast_ksic.csv` |
| R2 | `R2/e00_start.py` → `R2/exps.py E01 …`(외부 결합 `build_ext.py`, `build_feat_r2.py`, `build_word.py`; 코어 `r2core.py`, `r2lib.py`; `ablate.py`) → `R2/final_run_r2.py --mode dev\|final\|selftest\|future` → `R2/indep_check_final.py` | 챔피언 E57, 개발 원점 2010–2016, 최종시험 t0=2019·2020 | `<META>`, `build_ext.py`가 읽는 KOSIS(광업제조업조사·광공업생산지수·가동률·내수수출출하지수)·무역(`ksic_hs6_map.csv`, `hs6_import_yearly.csv`)·산업연관표 | `R2/cache_dev/`, `cache_2023/`, `ext_ksic_t0.parquet`, `ext_cell_t0.parquet`, `ext_link_*.csv`, `final_spec_r2.json`, `dev_result.json`, `final_result.json`, `future_forecast_r2.csv` |
| R3 (공식, 공개연도 변형) | 빌드 `R3/build_base_pub.py`, `build_early.py`, `build_ext3.py`, `build_ext_r3b.py`, `build_ext_r3c.py`, `build_ros_pubrule_r3d.py` → 실험 `R3/exps3.py --var pub E..`, `exps3b.py`, `exps3c.py`(추가 정의 `exps3*_more.py`, 코어 `r3core.py`, `r3lib.py`) → 감사 `audit_prefinal_r3.py`, `audit_trunc_r3d.py`, `audit_ros_pub_r3d.py`, `link_stats.py`, `diag_*.py` → `R3/final_run_r3.py --mode dev\|final\|selftest\|future` → `verify_final_r3.py`, `verify_final_r3d.py` | 챔피언 E40-auditD, 개발 원점 2010–2018, 최종시험 t0=2021(공개연도 2022–24) | `<META>`, `<EXT>`(r3oa·r3ros·r3kosis), KOSIS 광공업생산지수 | `R3/cache_pub/`, `cache_pub_final/`, `extB_*`·`extC_*`·`ext_*_t0_*.parquet`, `final_spec_r3.json`, `dev_result_r3.json`, `final_result_r3.json`, `preds/FINAL_R3_t0_2021.parquet`, `future_forecast_r3.csv`, `future_forecast_r3_ksic.csv`(그림 7) |
| R4 | `R4/t3_ntis_feats.py` → `R4/t3_exps.py` | NTIS 셀 피처 챔피언/도전자(R3 코어 import) | `<EXT>/ntis_*` | `R4/ntis_cell_pub.parquet`, 실험 기록 |
| VERIFY_R5 | `sens_t3.py pub`, `leak_t3.py`, `leak_t3_c2.py`, `ablate_t3.py`, `ci_sb_t3.py`(재현용 사본: `final_run_r2_origcache.py`, `final_run_r2_rebuilt.py`, `final_run_r3*.py`, `r2core/r2lib/r3core/r3lib.py`, `build_ext*.py`) | 레드팀 재검증 | 동결 예측·캐시, `<META>`, sqlite(ro) | `VERIFY_R5/*.json` |
| ESSENTIAL_R7 | `s01_build_conc_map.py`(+`s01b_build_conc_multi.py`) → `s02_build_alt_cache.py --alt ..` → `s03_build_alt_ext.py` → `t3_ess.py --var pub …`(`t3_ess_dev_snapshot.py`) → `s05_diag_coverage.py` → `s09_summarize.py` (공용 `conc_lib.py`, T3·N5 공용 사본) | 연계표 대체·5개국 제거 시험 | `R3/cache_pub/base_rows.parquet`, 연계표 xlsx | `rec_conc.parquet`, `conc_ksic_map.csv`, `conc_agreement*.{json,csv}`, `ESSENTIAL_R7/cache_*/`, `results/*.json`, `essential_table*.csv` |
| NONCIRC_R8 | `s01_build_caches.py --alt mg\|pcode` → `s02_ros.py` → `s03_ctx.py` → `nc_run.py --task mg --ctx …` → `s04_summarize.py`(+`s04b_s3pair.py`) → `s05_external.py` | 라벨 무관 단위(IPC 메인그룹)·대칭 2×2·외부 결과 정합 | `R3/cache_pub`(읽기만), `<EXT>/r3ros_work`, KOSIS·무역 | `NONCIRC_R8/cache_*/`, `results/`, `summary_r8.json` |
| TXT_R9 | `s01_txt_assign.py` → `s02_build_cache.py` → `s03_ros.py` → `s04_ctx.py` → `r9_run.py` → `s05_summarize_mg.py` → `s06_external.py` | TXT 셀 정의로 R8 시험 반복 | R3·R7·R8 산출물(읽기만) | `txt_assign.parquet`, `cache_txt_pub/`, `results/`, `summary_mg_r9.json`, `external_alignment_r9.csv`, `external_within_group_r9.csv` |

### 3.3 T4 특허 포기 (89개) — 살아 있는 KR 등록특허의 3년 내 등록료 불납 소멸 예측

| 라운드(폴더) | 진입 스크립트 | 역할 | 입력 | 출력 |
|---|---|---|---|---|
| 1R(최상위) | 추출 `extract.py`, `extract_biz.py`, `extract_family.py`, `extract_inv.py` → `build_feats.py`, `build_soc.py` → 실험 `run.py v0 …`, `s2exp.py`, `s2log.py`·`s2log_ens.py` (공용 `t4lib.py`, `hazard.py`, `soc.py`, `feat2.py`, `textfeat.py`, `versions.py`) → `make_spec.py` → `final_run.py --mode dryrun\|holdout\|future [--step ..]` → `stage3_verify.py`, `eval_final.py` · 감사 `audit/a1–a5`, `audit/m1–m2`, 2단계 수정 `s2/extract_family_dated.py` | 단계 `feats → frame_cls → frame_haz → fit_cls → fit_haz → fit_logit → fit_hgb → fit_base → pred → score`를 단계별 하위 프로세스로 실행. `final_model.py`는 '대체됨' 표시 | `registry_full.sqlite`(→ `cache/*.parquet`), `business_rights.sqlite`, `<META>`(패밀리) | `cache/patent`, `pay`, `right`, `owner`, `applicant` … `.parquet`, `final_models/run_<mode>/`(프레임·모형 pkl), `dryrun_result.json`, `holdout_result.json`, `stage3_verify.json`, `future_forecast.csv` |
| R2 (공식) | `R2/build_meta_cells.py`, `R2/make_links.py` → `R2/r2run.py frames O16` → `r2run.py champ` → `r2run.py exp E.. <mod>`(정의 `r2exp.py`, 피처 `r2ext.py`·`r2io.py`·`r2biz.py`, 하네스 `r2lib.py`, `r2boot.py`) → `R2/final_run_r2.py --mode dev\|final\|future` → `R2/audit/verify_final.py`, `diag_final*.py`, `check_future_consistency.py`; 트렌드 `make_trend_dev.py <exp>` | 개발 원점 O12·O14·O16·O18, 최종시험 기준일 L=2022-01-01 | 1R 모듈·캐시(읽기 전용 import), 산업연관표·KOSIS, 권리목록 | `R2/cache/`, `R2/runs/<O>/`, `R2/specs/`, `experiments.csv`, `final_spec_r2.json`, `dev_result.json`, `final_result.json`, `future_forecast_r2.csv`, `ext_link_*.csv`, `dev_ksic_risk_trend.csv` |
| R3 | 연계율 `R3/ext_link_A.py`, `ext_link_A2.py`, 탐침 `R3/probe/p1–p7` → `R3/r3run.py champ` / `exp X.. <mod>`(`r3exp.py`, `r3ext.py`, `r3lib.py`, `r3compose.py`) → `R3/make_final_spec_r3.py` → `R3/final_run_r3.py --mode dev\|future` | 새 최종시험 없음(등록원부 납부기록 끝 2025-08) | R2 하네스(읽기 전용 import), `<EXT>`(RoS·국세청·DART 등) | `R3/runs/<O>/`, `R3/runs_future/FUT/`, `final_spec_r3.json`, `dev_result_r3.json`, `future_forecast_r3.csv`, `future_vs_r2.json` |
| R4 | `R4/r4run.py init` → `exp …`(`r4exp.py`, NTIS 피처 `r4ext.py`, `r4lib.py`) | NTIS 정부R&D 기원 피처 챔피언/도전자 | `<EXT>/ntis_patent_link.parquet` | `R4/` 실험 json |
| VERIFY_R5 | `v1_labels.py` → `v2_retrain.py <구성요소>` → `v3_analysis.py`, `v4_reliance.py` | 레드팀: 원부에서 레이블 독립 재구성, 재학습 대조, 위약·민감도, 누수 의존도 | `registry_full.sqlite`(ro), 동결 `runs_final` | `VERIFY_R5/F22/`, json |
| ESSENTIAL_R7 | `e7_t4_x.py` → `e7_t4_run.py`, `e7_t4_e23check.py` (공용 `common/xwalk.py`, `common/firmrun.py` — N4·N7 필수성 시험도 이 공용 모듈을 쓴다) | KSIC 의존 피처를 연계표 배정으로 다시 계산해 동결 챔피언 R2-E03 재학습 | 연계표 xlsx, `kr_patents.parquet`, `R2/runs/<O>/` | `ESSENTIAL_R7/xcols/`, `runs/<VAR>/`, `res_*.json`, `xw_agreement*.{json,csv}` |

### 3.4 1라운드만 있는 트랙: T2 · T5 · T6

| 트랙 | 과제 | 진입 스크립트 | 입력 | 출력 |
|---|---|---|---|---|
| T2 출원추세 (27개) | (KSIC, 국가, 출원연도) 고유 출원 수 1–3년 앞 예측 | `build_panel.py`, 모형 `global_model.py`, `global_sub.py`, `ridge2.py`, `hier.py`, `flow.py`, `bottomup.py`, `scaled.py`, `stat_models.py <이름>`, `umsplit.py`, `adaptive.py`, `biascorr.py`, 결합 `ens.py`, 실험 `exp_a`–`exp_g.py`, `lab2.py`, 후보 `final.py`(v29) → 동결 `final_run.py --mode dryrun\|holdout\|future [--no-nowcast]` → `audit/holdout_recheck.py`, `audit/audit_data.py` (공용 `common.py`) | `<META>`, KOSIS(`exp_d/e*.py`) | `cache/`, `cache2/`, `dryrun_result.json`, `dryrun_preds.parquet`, `holdout_result.json`, `holdout_forecast.csv`, `holdout_preds.parquet`, `future_forecast.csv`, `future_completeness_factors.csv` |
| T5 기술거래 (81개) | KR 특허 권리이전 예측(권리자 매각률 × 특허 점수의 2단 계층모형) | 파이프라인 `s10_extract.py` → `s11_inv.py` → `s20_panel.py` → `s21_extra.py` → `s23_unit.py` → `s24_techexit.py` → `s50_actfix.py` → `s51_ownerattr2.py` (공용 `common.py`, `evalkit.py`, `run.py`, `v2lib.py`; 탐색 `s00`–`s03`, 추가 피처 `s22`·`s25`–`s27`·`s30`·`s40`·`s52`–`s54`, 모형 `owner_model*.py`, `tune.py`, `blend.py`) → 동결 `final_run.py --mode dryrun\|holdout\|future [--stage build\|fit\|all] [--rebuild]` → `holdout_check/h02_recompute.py`, `h03_diag.py`, `audit/h01_hasowner_check.py` | `registry_full.sqlite`, `<META>`(`s26_global.py`), `DART\dart_company.sqlite`(`s27_dart.py`) | `_work_<mode>/`(실행기가 파이프라인 스크립트를 복사해 캐시를 새로 만드는 폴더. 저장소의 `_work_*/*.py`가 그 사본이며 `_work_future/s20_panel.py`만 원본과 다르다), `dryrun_result.json`, `holdout_pred.parquet`, `holdout_result.json`, `future_forecast.csv` |
| T6 산업성장 (22개) | KSIC 5자리 산업의 1년 앞 성장 예측(광업제조업조사 지표, KSIC 10·11차 연결) | `01_patent_agg.py` → `02_registry_agg.py` → `03_build_panel.py` → `04_patent_features.py` → `10_v0_baseline.py` → `20_run.py v..`(정의 `versions.py`, `models.py`, `t6lib.py`) → `30_screen.py`, `40_robust.py`, `50_trend_draft.py` → `70_run2.py` → 동결 `final_run.py --mode dryrun\|holdout\|future` · 감사 `audit/a01–a06`, `h01_independent_check.py`, `h02_trunc_diag.py` | `<META>`, `registry_full.sqlite`, `작업데이터/kosis_광업제조업_*.csv`, `외부데이터/경기지표·재무·제3축/*.csv` | `cache/`, `dryrun_result.json`, `holdout_result.json`, `future_forecast.csv`, `*_preds*.csv`, `diag/` |

### 3.5 N 트랙(2–6라운드 신설)

| 트랙(파일 수) | 생긴 라운드 | 진입 스크립트(→ 순서) | 주요 입력 | 주요 출력 |
|---|---|---|---|---|
| N1 기술점유·무역 → 산업 고용 조기경보 (45) | R2 | 집계 `a01_patent_agg2.py` … `a06_link_summary.py` → `b01_panel.py` → `s01_baselines.py` → `s02_challenges.py <batch>` → `s03_report_tables.py`, `s_inc.py`, `sD_export.py`, `d01_univar.py` (엔진 `engine.py`, 공용 `n1lib.py`) → `final_run_r2.py --mode dev\|final\|future` → `r01_verify_final.py`, `r02_posthoc_future.py`. `c02_fetch_final_exports.py`는 🔒 최종시험 때 1회만(UN Comtrade). `pilot_src/`는 `_ideas_R2/industry/`와 같은 파일 26개 | `<META>`, `registry_full.sqlite`, 산업연관표, KOSIS, 무역(`<EXT>`) | `cache/`, `R2/final_dev_result.json`, `final_result.json`, `R2/increment_log.csv`, `ext_link_*.csv` |
| N2 특허 브리지 산업 나우캐스트 (18) | R2 | `n00_link.py` → `n01_dart_collect.py`(OpenDART, 키 필요) → `n02_fin.py` → `n03_alloc_pit.py` → `n04_panel.py` → `n05_fetch_export.py` → `n10_start.py` → `n11`–`n16_batch*.py`(+`n12_rebase.py`) → `n20_controls.py`, `n21_links.py` (공용 `n_lib.py`) → `final_run_r2.py --mode dev\|final\|future` → `verify_final_r2.py` | `registry_full.sqlite`, `business_rights.sqlite`, DART, KOSIS, 무역, `ksic_hs6_map.csv` | `cache/`, `R2/experiments.csv`, `R2/dev_frozen_check.json`, `final_result.json`, `future_nowcast_2025.csv` |
| N3 기술-수출 괴리 HS6 수출 예측 (11) | R2 | `build_panel.py` → `n3_build.py` → `n3_run.py X\|M <exp>` (`n3_lib.py`, `common_n3.py`) → `n3_diag.py`, `n3_robust.py`, `n3_links.py` → `final_run_r2.py --mode dev\|prereg\|final\|future` → `verify_final.py`; 보충 수집 `fetch_5c_exports.py` | `<META>`, Comtrade 5개국 수출(`<EXT>`), KOSIS, IO-KSIC-HS 대조표 | `cache/`, `dev_result_r2.json`, `prereg_t2021_{X,M}.csv`, `final_result.json` |
| N4 특허 행태 → 기업 폐업 조기경보 (21) | R4 (아이디어 I2b) | `n4_01_panel.py` → `n4_02_bundles.py` → `n4_03_exp.py E..` → `n4_04_rl_leak_check.py`, `n4_05_posthoc.py`, `n4_06_patent_value.py` → `audit_r4.py` → `final_run_r4.py --mode dev\|final\|future` → `verify_final_r4.py` (공용 `n4lib.py`, `n4feat.py`) · R5 `VERIFY_R5/v1`–`v8` · R7 `ESSENTIAL_R7/e7_n4.py`, `ext_validity_firm.py` | `registry_full.sqlite`, `business_rights.sqlite`, `kr_patents.parquet`, `CORP_APPLICANT.txt`, `<EXT>`(국세청·국민연금·DART·NTIS) | `cache/base_dev.parquet`, `cache/feat_dev.parquet`, `cache/<mode>_*`, `final_spec_r4.json`, `final_dev_check_r4.json`, `final_result_r4.json`, `future_pred_r4.parquet`, `future_ksic_risk_r4.csv` |
| N5 표준특허(SEP) 선언 레이더 (14) | R4 (아이디어 I5) | `n5_build.py [KSIC]`(기본 26429) → `n5_eval.py E..` → `n5_info.py` → `final_run_r4.py --mode dev\|final --confirm-final\|future` → `final_verify_r4.py` (공용 `n5lib.py`) · R5 `VERIFY_R5/v0`–`v3` · R7 `ESSENTIAL_R7/n5_panels.py --pop ..` → `n5_fit.py` → `n5_summarize.py` | `kr_patents.parquet`, `<EXT>`(ETSI SEP 연결, NTIS, OpenAlex) | `cache/`, `final_spec_r4.json`, `final_dev_check_r4.json`, `final_result_r4.json`, `final_preds_r4.parquet`, `future_scores_r4.parquet`, `ESSENTIAL_R7/essential_table.csv` |
| N6 정부R&D 수행기업의 첫 특허 (11) | R5 (아이디어 I8) | `n6_text.py` → `n6_features.py dev` → `n6_exp.py` → `n6_robust.py` → `final_run_r5.py --mode dev\|final\|future` (공용 `n6_lib.py`) → `audit_r5/verify_final_r5.py`, `audit_r5/placebo_linked_r5.py`(`audit_r5/pre_audit/`는 감사 전 사본) · R7 `ESSENTIAL_R7/r7_n6.py` | `<EXT>`(NTIS 과제·KSIC 텍스트 대응, 국세청), `kr_patents.parquet` | `cache/panel_dev.parquet`, `final_spec_r5.json`, `final_dev_check_r5.json`, `final_result_r5.json`, `final_preds_r5.parquet`, `future_scores_r5.csv`, `future_ksic_r5.csv` |
| N7 특허기업의 정부R&D 첫 수행 (24) | R5 (아이디어 I7) | `n7_01_panel.py` → `n7_02_feat.py` → `n7_03_exp.py` → `n7_04_rl_leak_check.py`, `n7_05_posthoc.py` → `n7_06_audit.py`(최종 전 감사) → `final_run_r5.py --mode dev\|final\|future` → `n7_07_verify.py`, `n7_08_placebo_diag.py` (공용 `n7lib.py`, `n7feat.py`) · R6 `VERIFY_R6/w1`–`w9`(공용 `w_common.py`) · R7 `ESSENTIAL_R7/ky_xw.py`, `e7_n7.py`, `e7_n7_final_posthoc.py` | `registry_full.sqlite`, `business_rights.sqlite`, `kr_patents.parquet`, `<EXT>`(NTIS·국민연금·국세청) | `cache/feat_dev.parquet`, `final_spec_r5.json`, `final_dev_check_r5.json`, `final_result_r5.json`, `final_test_pred_r5.parquet`, `future_pred_r5.parquet`, `future_ksic_entry_r5.csv`, `VERIFY_R6/cache/`, `w8_placeboB.json` |
| N8 특허 분쟁 표적 레이더 (14) | R6 (아이디어 J2) | 탐침 `p00_probe.py`, `p01_claims_probe.py`, `p02_claims_auc.py` → `b01_build_dev.py` → `e_run.py xgb\|logit <EXP>` → `r_report.py`, `s6_placebo.py` → `final_run_r6.py --mode dev --part 1` → `--part 2` → (`--mode final`) → `v01_final_independent.py`, `v02_final_placebo.py`, `v03_final_sensitivity.py` (공용 `n8lib.py`) · R7 `ESSENTIAL_R7/e7_n8.py`, `e7_n8_seeds.py` | `registry_full.sqlite`(봉인 추출: 사건 < 2023-01-01), `<META>`, KOSIS, `ksic_hs6_map.csv`, `<EXT>`(RoS·KPSS·ETSI·NTIS) | `cache/panel_dev.parquet`, `cache/devcheck_part1.json`, `final_spec_r6.json`, `final_dev_check_r6.json`, `final_result_r6.json`, `final_preds_r6.parquet`, `future_scores_r6.parquet`, `future_ksic_r6.csv` |
| N9 정부R&D 과제의 특허 성과 예측 (16) | R6 (아이디어 J1) | `n9_text.py` → `n9_build.py dev` → `n9_textclf.py dev` → `n9_exp.py R..\|C..\|report` → `n9_freeze.py` → `n9_robust.py …` → `final_run_r6.py --mode dev --part 1` → `--part 2` → (`--mode final`) → `n9_final_audit.py` (공용 `n9_lib.py`) · R7 `ESSENTIAL_R7/r7_concord.py`, `r7_agree.py`, `r7_n9_alt.py`, `r7_n9_alt_p1b.py` → `r7_n9_run.py` → `r7_summarize.py`, 정의 시험 `r7_n9_yalt.py` | `<META>`, `<EXT>`(NTIS 과제·성과·연결), KOSIS, KSIC 해설서 json | `cache/`(`champ_state.json`, `devcheck_*`), `experiments.csv`, `final_spec_r6.json`, `final_dev_check_r6.json`, `final_result_r6.json`(그림 9), `final_preds_r6.parquet`, `future_scores_r6.csv`, `future_ksic_r6.csv`, `ESSENTIAL_R7/summary_dev.csv` |

아이디어 이름(I2b, I5, I7, I8, J1, J2)은 각 트랙 공용 모듈 머리말(`n4lib.py` 'N4 (I2b)' 등)에 적힌 것이다.

### 3.6 아이디어 파일럿 `_ideas_R2`–`_ideas_R8` (122개)

| 폴더 | 내용(머리말 기준) | 이어진 트랙 |
|---|---|---|
| `_ideas_R2/industry/` (26) | 산업 단위 파일럿 `p01`–`p52`(수출·수입·KOSIS·산업연관표), `ilib.py` | N1(`N1/pilot_src/`가 같은 파일) |
| `_ideas_R2/firm/` (20) | 기업 연결(DART), 고성장·부실화, 기술 구매기업, 특허 포기, 산업 나우캐스트 `f30`–`f33`, `evalkit.py` | N2(`n00_link.py`가 `f00_link.py`를 복사·수정) |
| `_ideas_R2/global/` (15) | 5개국 패널, HS6 수입·수출(G1·G2), T2 재도전(G3) | N3(추정: `build_panel.py` 머리말이 같다) |
| `_ideas_R3/` (11) | I1 잠재 첫 출원 기업, I2 특허기업 소멸·폐업(I2', I2b), I3 고용 고성장, I4 첫 해외출원, I5 SEP 선언 | N4(I2b), N5(I5) |
| `_ideas_R4/` (15) | I6 공공기술 사업화, I7 특허기업 정부R&D 첫 수행, I8 정부R&D 기업 첫 특허 | N7(I7), N6(I8) |
| `_ideas_R5/` (9) | J1 정부R&D 과제 특허 성과, J2 특허 분쟁 표적, J3 피합병(M&A) 표적 | N9(J1), N8(J2) |
| `_ideas_R6/` (10) | K1 국내 최초출원 → 해외 공개, K2 해외 → KR 공개(+PCT 판), K3 탐색 | 없음 |
| `_ideas_R7/` (8) | KRX KIND 상장·상장폐지 수집, IPO 레이더 파일럿, 업종 외부 타당성 | 없음(외부 타당성 결과를 숫자 고정표가 인용) |
| `_ideas_R8/` (8) | P1 기술의 산업 간 이동 레이더, P2 연계표 그룹 안 세세분류 분화, S-TXT 배정 | 없음(결과 json을 숫자 고정표가 인용) |

공통 파일 패턴: `*_common.py`(경로·봉인 상수·평가), `*_build.py`(학습표), `*_eval.py`(사전 등록 순차 실험 M0 → M1 → …; `_ideas_R4/r4_eval.py` 머리말의 채택 기준은 '평균 Δskill ≥ +0.003 그리고 원점 60% 이상 우세'), `*_eval_spec.py`(사전 고정 명세), `*_robust.py`(사후 점검).

### 3.7 `_shared_report/` (52개) — 트랙 공동 분석(7–12라운드)

| 폴더 | 라운드 | 내용 | 진입 스크립트 → 주요 산출 |
|---|---|---|---|
| `audit_r7/` | R7 | 연계표 해상도와 제공 라벨 비교(공고문 부합성 감사 보조) | `concordance_resolution_check.py` → `concordance_resolution_check.json` |
| `label_validity/` | R7 | 제공 KSIC 라벨의 외부 타당성: 특허 단위(연계표 일치), 기업 단위(국민연금·DART 업종), 산업 단위(KOSIS·무역) | `lv_common.py`, `s1_patent_agree.py`, `s2_firm_agree.py`, `s3_industry.py` → `lv1_*.csv`, `lv2_*.csv`, `cache_kr_cw.parquet` |
| `fusion_advantage/split34/` | R8 | 연계표가 한 코드로 묶는 34개 KSIC의 그룹 안 분화 | `s1_panel.py` → `s2_decomp.py` → `s3_forecast.py` → `s4_figs_robust.py` → `s5_cases.py` → `t5_cases.csv`, `fig_smallmultiples_long.csv`(그림 4) |
| `fusion_advantage/map5/` | R8 | 50 KSIC × 5개국 기술 지형도(RTA, WIPO 커버리지 보정, 선행-후행, 수출 사분면, 대시보드) | `m1_build_counts.py` → `m1b_field_counts.py` → `m2_trend_map.py` → `m3_leadlag.py`·`m3b`·`m3c` → `m4_industry_quadrant.py`·`m4b` → `m5_dashboard.py` → `m6_figures.py` → `m7_supplement.py` → `m2_ksic_profile.csv`, `fig1_rta_map_P3.csv`(그림 5), `fig2_quadrant_tech_export.csv` |
| `noncirc_n9n6_R9/` | R9 | N9·N6 순환 없는 검증 | `n9_a_build.py` → `n9_b_labelcols.py` → `n9_c_run.py`; `n6_features_r9.py` → `n6_a_build.py` → `n6_b_run.py`; `d_eval.py` → `results.json`, `results_table.csv` (공용 `r9_common.py`) |
| `fusion_advantage/c5_R10/` | R10 | 해외 문서 대체 배정(CW·TXT·FREE-IPC·PKR-IPC)으로 5개국 정보의 고유성 시험 | `c1_assign.py` → `c2_rta.py` → `c3_build_cache.py` → `c3_t3_run.py` → `c4_p1.py` → `c5_t3_summary.py` → `alt_assign/`. `t3cache_hyb_*/build_panel.py`는 T3 동결 빌더 사본 |
| `label_origin_R10/` | R10 | 제공 라벨 생성 방식(기업 업종 일치 시험의 순환 여부), 패밀리 연결 재구축 | `lo_fam.py`, `lo_prep.py` → `lo_tests.py`, 사후 `lo_posthoc.py`·`lo_posthoc2.py` → `res_p1.csv`(그림 3) |
| `label_origin_R10/foreign_public_R11/` | R11 | 해외 공개 라벨과 제공 해외 라벨의 출원인 등록업종 일치 | `fp_prep.py` → `fp_tests.py`, 사후 `fp_posthoc.py`·`fp_posthoc2.py` |
| `label_origin_R10/confirm_R12/` | R12 | 근거 A 확인 시험(공개 최강 FREE-IPC 대비, 사전 등록) | `r12_prep.py` → `r12_reg.py` → `r12_tests.py` → `res_main.csv`(그림 3) |
| 최상위 | R4–R10(추정) | NTIS 기술통계 표(예측 성적과 무관) | `build_ntis_tables.py` → CSV → `build_ntis_md.py` → `ntis_tables.md` |

## 4. `_shared/` — 공용 메타 빌드와 외부데이터

### 4.1 `_shared/build_meta.py` — 제공 데이터를 메타 parquet로

| 항목 | 내용 |
|---|---|
| 실행 | `python 분석/모델경쟁_20260927/_shared/build_meta.py` |
| 입력 | `<ROOT>\공모전데이터\융합데이터_*\*.csv` 50개(확장자는 csv지만 탭 구분, utf-8-sig, 합계 29.5GB) |
| 처리 | 16개 열만 읽고(요약·대표청구항 제외) 영문 이름으로 바꾼다: `appl_no`, `appl_date`, `appl_year`, `rgst_no`, `rgst_date`, `pub_date`, `title`, `cc`, `applicant`, `main_ipc`, `ipc_all`, `main_cpc`, `family`, `prio_cc`, `prio_date`, `ksic`. 40만 행 단위로 읽고, 5개 프로세스 병렬 |
| 출력 | `<MC>\_shared\meta\<원본 파일 이름>.parquet`(파일별 1개, 이미 있으면 건너뜀). `_shared/meta/`는 `.gitignore`로 제외 |

### 4.2 `_shared/ext/` — 외부데이터 수집·가공(77개 + `demand_work/` 6 + `r3oa_work/` 13)

산출물은 `<EXT>`와 그 아래 작업 폴더(`ntis_B_work/`, `ntis_T_work/`, `r3nts_work/`, `r3ros_work/`, `quality_raw/` 등)에 쓴다. 수집 스크립트는 네트워크와(일부) API 키가 필요하며, 머리말 규칙은 '로그인 없는 공개 데이터만, 요청 간 1–3초·지수 백오프·한도 신호 시 즉시 중단, 키 값 출력·저장 금지, 제공 데이터 내용은 외부로 보내지 않음(로컬 결합)'이다. 의미 검정(`*_test*.py`)은 개발 구간 자료만 쓴다.

| 접두사 | 데이터 | 수집 | 가공·검정 | 대표 산출 |
|---|---|---|---|---|
| `firm_` | 국민연금 가입 사업장(data.go.kr), DART 재무, KRX KIND 상장폐지, 벤처기업명단 | `firm_nps_fetch.py`, `firm_misc_collect.py` | `firm_nps_link.py`, `firm_nps_ksic_bridge.py`, `firm_nps_patw.py`, `firm_dart_build.py`, `firm_nps_test_lapse.py` | `firm_nps_wkpl_<YYYYMM>.parquet`(추정: 수집 산출), `firm_nps_applicant_link.parquet`, `firm_nps_firm_panel.parquet`, `firm_nps_ksic50_panel.parquet`, `firm_dart_fin_panel.parquet`, `firm_nps_link_rates.csv` |
| `ntis_` | NTIS 국가R&D 과제·성과(Open API) | `ntis_B_collect.py [분]`, `ntis_rpatent_collect.py`(종료코드 0=시간 소진·2=완료·3=중단), `ntis_B_probe*.py` | `ntis_B_build_raw.py`, `ntis_B_build_ksic.py`, `ntis_B_ksic_dict.py`, `ntis_B_score_dict.py`, `ntis_rpatent_build.py`, 트랙별 검정 `ntis_T1_test.py`·`ntis_T3_test.py`·`ntis_T3_champ.py`·`ntis_T4_test.py` (공용 `ntis_common.py`, `ntis_B_common.py`, `ntis_T_common.py`) | `ntis_project_raw.parquet`, `ntis_project_year_agg.parquet`, `ntis_project_ksic_year.parquet`, `ntis_project_ksic_text_map.parquet`, `ntis_rpatent_raw.parquet`, `ntis_patent_link.parquet`, `ntis_patent_flag.parquet`, `ntis_link_rates.csv` |
| `quality_` | OECD 특허통계, WIPO 공개 건수, KPSS 특허가치, ETSI SEP, KIPRIS Plus | `quality_fetch_oecd.py`(키 불필요), `quality_kipris_util.py`(키 로더) | `quality_build_etsi_link.py`, `quality_build_kpss_link.py`, `quality_build_wipo_compare.py`, `quality_wipo_validate.py`, `quality_test_*.py` | `quality_etsi_sep_link.parquet`, `quality_kpss_family_link.parquet`, `quality_wipo_ipc_field_concordance.csv`, `quality_provided_cc_year_wipo35.parquet`, `quality_oecd_patents_wipo35.parquet` |
| `r3kip_` | KIPRIS Plus 인용·피인용(표본) | `r3kip_collect.py`, `r3kip_o18_collect.py`(누적 호출 상한) | `r3kip_sample.py`, `r3kip_o18_sample.py`, `r3kip_parse.py`, `r3kip_build.py`, `r3kip_o18_build.py` (`r3kip_util.py`) | `r3kip_sample.parquet`, `r3kip_citing_edges.parquet`, `r3kip_patent_fwd_asof.parquet`, `r3kip_ksic_indicators.parquet` |
| `r3kosis_` | KOSIS 품목편(연간)·월별(생산능력·가동률·국내공급) | `r3kosis_collect_items_annual.py`, `r3kosis_collect_monthly.py` (`r3kosis_lib.py`) | `r3kosis_build_items.py`, `r3kosis_build_chain.py`(분류 개정 연쇄 연결), `r3kosis_build_monthly_link.py`, 선행 검정 `r3kosis_lead_test*.py` | `r3kosis_ksic50_items_annual.parquet`, `r3kosis_ksic50_items_chain.parquet`, `r3kosis_ksic50_capacity_util_annual.parquet`, `r3kosis_lead_test_pub_panel.parquet` |
| `r3nts_` | 국세청 사업자 상태조회 | `r3nts_01_targets.py` → `r3nts_02_collect.py`(`r3nts_keytest.py`) | `r3nts_03_build.py` | `r3nts_status.parquet`, `r3nts_applicant_status.parquet`, `r3nts_link_by_ksic.csv`, `r3nts_link_by_year.csv` |
| `r3oa_`(`r3oa_work/`) | OpenAlex IPC 메인그룹 논문 수 | `r3oa_04_fetch.py`, `r3oa_07_fetch2.py`(OpenAlex polite pool: 연락처 이메일이 코드에 있음 — 점검표 참고) | `r3oa_01_cells.py`, `r3oa_02_titles.py`, `r3oa_03/08_validity*.py`, `r3oa_05/10_*test*.py`, `r3oa_06/09_link_ksic*.py`, `r3oa_11_features.py`, 검색어 사전 `r3oa_terms.py`·`r3oa_terms_b.py` | `r3oa_mg_year_ALL.parquet`, `r3oa_mg_year_company.parquet`, `r3oa_mg_t0_features.parquet`, `r3oa_cell_link*.csv` |
| `r3ros_` | Reliance on Science(논문-특허 인용) + OpenAlex 분야 | `r3ros_04_oafield.py`(OpenAlex 표본 조회, 연락처 이메일 있음) | `r3ros_01_family.py`, `r3ros_02_build.py`, `r3ros_03_test.py`, `r3ros_05_mapeval.py` | `r3ros_patent_sci.parquet`, `r3ros_work/`, `r3ros_link_*.csv`, `r3ros_ipc_mg_openalex_subfield.parquet` |
| `science_` | OpenAlex KSIC 키워드 논문, OECD ANBERD·ECOS R&D | `science_fetch_openalex.py`, `science_fetch_openalex_extra.py`, `science_fetch_rd_stats.py`(키) | `science_linkage_stats.py`, `science_meaning_test.py`, `science_meaning_test_extra.py` | `science_openalex_ksic_country_year.parquet`, `science_openalex_ksic_extra_year.parquet`, `science_openalex_country_total_year.parquet`, `science_oecd_anberd_isic4.parquet` |
| `trade_` | UN Comtrade 한국 HS6 수출 단가, ISO 표준, ETSI IPR, 반덤핑(TTBD) | `trade_02_fetch_kr_export_uv.py`(키 불필요) | `trade_01_patcnt.py`, `trade_03_iso_ksic.py`, `trade_05_etsi_parse.py`, `trade_06_etsi_link.py`, `trade_08_ttbd.py`, 검정 `trade_04_tests.py`·`trade_07_tests2.py` (`trade_lib.py`) | `trade_patcnt_ksic_year_cc.parquet`, `trade_etsi_sep_patents.parquet`, `trade_etsi_sep_link.parquet`, `trade_ics_ksic_map.csv`, `trade_iso_ksic_year.parquet` |
| `demand_work/` | ECOS 품목 물가, NBER-CES(미국 제조업) | `demand_c01_ecos_prices.py`(`demand_keys.py`) | `demand_c02_map.py`, `demand_c03_test.py`, `demand_c04_nberces.py`, `demand_c05_decomp.py` | `demand_ksic_price_panel_1996_2021.parquet`, `demand_ksic_ecos_price_map.csv` |

`R2global_*`, `R2ind_*`, `R2N2_*`, `R2N3_*` 같은 무역 수집본(예: `R2global_kr_export_hs6.csv`)은 `_ideas_R2/global/fetch_kr_export.py`, `_ideas_R2/industry/c01_fetch_exports.py`, N2 `n05_fetch_export.py`, N3 `fetch_5c_exports.py`가 `<EXT>`에 쓴 것으로 보인다(추정: 파일 이름 접두사와 소비 코드 경로 기준).

## 5. `_workflows/` — 오케스트레이션 스크립트

### 5.1 성격과 실행 방식

- 모두 `export const meta = { name, description, phases }`로 시작하고, 본문에서 `agent(프롬프트, { label, phase, schema, effort })`, `parallel([...])`, `pipeline(목록, 단계1, 단계2)`, `phase(이름)`, `log(문장)`, `args`를 선언 없이 쓴다. `require`·`import`가 없다.
- 따라서 node로 직접 실행하는 파일이 아니라, 이 전역 함수들을 넣어 주는 에이전트 워크플로 실행기(Claude Code의 워크플로 기능)에서 돌리는 스크립트다(추정: 실행기 이름은 코드에 없다).
- 각 `agent` 호출은 하위 에이전트 하나에게 일을 맡긴다. 프롬프트에 경로(`<ROOT>` 절대경로), 절대 규칙(원자료·동결 파일 읽기만, 지정 폴더에만 쓰기, sqlite `mode=ro`, 공식 성적 불변, 키 값 출력 금지, 메모리 4GB·`n_jobs ≤ 2`·스크립트 9분), 응답 JSON 스키마가 들어 있다.
- 산출물(라운드 보고서 `R*_결과_*.md`, 원고, 원장)은 하위 에이전트가 지정 폴더에 쓴다. 이 파일들은 저장소에 없다.

### 5.2 파일별 역할

| 파일 | meta.name | 하는 일 | 쓰는 위치(프롬프트 기준) |
|---|---|---|---|
| `r6.js` | model-competition-r6 | N8·N9 구축 → 최종시험(N8 t0=2022, N9 t=2022, 1회), N7 국민연금 연결키 재검증, 7라운드 아이디어 → 종합 | 각 트랙 폴더, `_ideas_R6/`, `R6_결과_20261004.md` |
| `r7.js` | r7-contest-fit-essentiality | 공고문 부합성 감사, 9개 트랙 필수성(제거·대체·5개국), 라벨 외부 타당성, 데이터 확보 → 종합 | 각 트랙 `ESSENTIAL_R7/`, `_shared_report/label_validity/`, `_ideas_R7/`, `R7_결과_20261005.md` |
| `r8.js` | r8-fusion-advantage | 순환 없는 필수성(T3·T1), 34개 산업 분화, 5개국×50산업 지형도, 고유 장점 아이디어 → 종합 | `T3·T1/NONCIRC_R8/`, `_shared_report/fusion_advantage/split34·map5/`, `_ideas_R8/`, `R8_결과_20261005.md` |
| `r9.js` | r9-txt-threat | 두 번째 대체안(KSIC 해설서 텍스트) 위협 검증(T1·기업 업종·T3), N9·N6 순환 없는 검증 → 종합 | `T1·T3/TXT_R9/`, `_shared_report/noncirc_n9n6_R9/`, `R9_결과_20261005.md` |
| `r10.js` | r10-5country-and-label-origin | 5개국 라벨 고유성, 라벨 생성 방식(순환 여부) → 종합 | `_shared_report/fusion_advantage/c5_R10/`, `_shared_report/label_origin_R10/`, `R10_결과_20261005.md` |
| `r11.js` | r11-verify-and-report-draft | 해외 공개 라벨 검증 + 숫자 고정표·그림 → 장별 원고 5개 → 편집(보고서 초안) | `label_origin_R10/foreign_public_R11/`, `보고서_20261012/facts/`, `figs/`, `drafts/`, `manuscript_v1.txt` |
| `r12.js` | r12-confirm-and-revise | 근거 A 확인 시험(사전 등록) + 그림 결함 수정 → 원고 v2 | `label_origin_R10/confirm_R12/`, `보고서_20261012/figs/`, `manuscript_v2.txt` |
| `r13.js` | r13-figures-print-size | 그림을 최종 인쇄 크기(폭 17cm, 글자 7pt 이상)로 다시 그림 → docx 재조판·쪽수 확인(`FIG_NATIVE=1 node md2docx.js …`, `render.ps1`) | `보고서_20261012/figs/`(원본은 `figs/_v2/`로 보관), `보고서_초안_v3.docx`, `manuscript_v3.md` |
| `verify_loop.js` | verify-loop | 검증 무한 루프(6절): 바퀴마다 계획 → 실행(최대 4개 병렬) → 반박 → 원장 기록. `STOP.txt`가 있거나 30바퀴가 되면 멈춘다 | `검증루프_20261012/Cxx/Cxx-Tn/`, `LEDGER.jsonl`, `FINDINGS.txt`, `ERRATA.txt` |
| `verify_loop2.js` | verify-loop-2 | 위와 같은 루프 + 재개 기능. `args = { start, history, firstPlan }` 또는 `{ start, resumeFile, resumeDone, resumeVerDone }` | 같음 |
| `final_polish.js` | final-polish | 원고 4구간 병렬 편집(숫자 대조기 `numcheck.py` PASS 필수) → 구간별 반박 검증·수선 → 조립·빌드(md2docx + render.ps1, PyMuPDF로 쪽 배치 점검, 20쪽) → 독립 최종 감사 | `보고서_20261012/drafts/polish/`(`.gitignore`로 제외, `numcheck.py`도 저장소에 없음) |

2–5라운드를 묶은 워크플로 파일은 저장소에 없다(다른 방식으로 진행했거나 보관하지 않은 것으로 추정).

## 6. `분석/검증루프_20261012/` — 바퀴·작업 구조와 명명 규칙

### 6.1 한 바퀴가 도는 방식(`verify_loop*.js` 기준)

1. **계획**: 원장(`LEDGER.jsonl`·`FINDINGS.txt`·`ERRATA.txt`)을 읽고 보고서 신뢰성에 영향이 큰 순서로 작업 3–4개를 고른다. 작업 종류는 `data_audit`(데이터 품질·편향 감사), `result_audit`(원고 숫자·결과 오류 감사), `track_reverify`(기존 트랙 재검증), `new_idea`(새 과제), `data_acquire`(데이터 확보).
2. **실행**: 작업마다 하위 에이전트가 `Cxx\Cxx-Tn\`에만 쓴다. 실행 전에 판정 기준을 `PREREG.txt`에 적고, 새 데이터는 `data_check.txt`에 (a) 공식 총량 대조 (b) 분포·결측·중복 (c) 연결 선택 편향 (d) 시점 정합을 남긴다. 발견은 `error`·`warning`·`ok`·`new_result`로 나눈다.
3. **반박**: `error`와 `new_result` 발견마다 별도 에이전트가 원천 파일에서 다시 계산해 반박을 시도하고, `verify_<k>\`(k = 그 작업의 반박 대상 발견 순번, 0부터)에만 쓴다. 불확실하면 '확인 안 됨'으로 처리한다.
4. **기록**: 반박 검증에서 확인된(`confirmed`) `error`·`new_result`와 주요 `warning`은 `FINDINGS.txt`, 원고 수정 제안(확인된 것만)은 `ERRATA.txt`, 바퀴 요약은 `LEDGER.jsonl`에 덧붙인다.

원장·PREREG·data_check·캐시·결과 csv/json은 저장소에 없다(코드만 있음).

### 6.2 폴더 구조

```text
분석/검증루프_20261012/
  C01/ … C20/                      바퀴(cycle). C20은 T1–T3만 있고 verify_* 폴더가 없다
    Cxx-T1/ … Cxx-T4/              작업(task). id 형식은 워크플로 스키마의 예 'C03-T1'과 같다
      scripts/                     실행 스크립트(C01-T1·C01-T3은 작업 폴더 바로 아래)
        <하위 폴더>/               트랙 코드 사본으로 다시 돌리는 하위 실행(t3run, n4, n6, n7, n8, n9, t4, t4mirror, map, n5, t1, t3, lo)
      verify_0/, verify_1/, …      반박 검증(번호 = 반박 대상 발견 순번)
      verify/                      번호 없는 검증 폴더(C17-T2, C18-T1, C19-T2, C19-T3; 실행 단계 자체 검증으로 추정)
      build_test/, verify_build/   보고서 빌드 시험(md2docx.js·render.ps1 사본, 그림 스크립트 사본)
      regen/                       그림 재생성 시험(C06-T2: _style.py, make_*.py 사본)
```

규모: 79개 작업, 파일 1,358개(`.py` 1,338, `.js` 10, `.ps1` 10). `scripts/` 아래 890개, `verify_<k>/` 아래 418개.

### 6.3 스크립트 이름 규칙

| 위치 | 이름 | 뜻 |
|---|---|---|
| `scripts/` | `s<번호>[a-z]_<무엇>.py` (557개, 예: `s0_probe.py`, `s1_build.py`, `s3b_bodytokens.py`) | 실행 단계. 번호 순서로 돈다. `s0`·`s00`은 대개 구조·형식 탐침(결과 지표 계산 없음), `s1`부터 구축·재계산·판정. 같은 번호에 붙은 a·b·c는 하위 단계나 보완 |
| `scripts/` | `lib<바퀴>.py`, `c<바퀴>lib.py`, `c<바퀴>common.py`, `common.py`, `wlib.py` | 그 작업의 공용 경로·로더·평가 함수. 쓰기는 작업 폴더로 제한 |
| `scripts/` | `guard*.py` | 쓰기 가드: 작업 폴더 밖으로 쓰면 예외(예: C10-T1 `guard.py`, C17-T1 `guard17`) |
| `scripts/` | `p*_`, `x0_peek.py`, `e0_peek.py`, `*_schema.py` | 사전 점검·형식 확인(PREREG 전, 결과 없음) |
| `scripts/` | `final_run_r*.py`, `run_*.py`, `*core.py`, `*lib.py` 사본 | 트랙 동결 실행기·코어의 사본으로 재학습·재채점(경로만 바꾼 사본임을 머리말에 적음) |
| `scripts/` | `assemble.py`, `build_*.py`, `s0_sha.py`, `patch_*.py` | 결과 취합, 표·지도 작성, 입력 SHA-256 기록, 스크립트 1회성 수정 |
| `verify_<k>/` | `v<번호>[a-z]_<무엇>.py` (예: `v1_recompute.py`, `v0_refute.py`) | 반박 검증 재계산. 대개 작업자 캐시·라이브러리를 쓰지 않는 독립 구현이며 원천(sqlite `mode=ro`, parquet)을 직접 읽는다 |
| `verify_<k>/` | `r<번호>_<무엇>.py` (예: `r1_regpath.py`) | 반박 검증(머리말 '반박 검증 r1') |
| `verify_<k>/` | `w*_`, `a*_`, `indep.py` 등 | 보조 재계산·탐색(추정) |

### 6.4 바퀴별 작업 요지(각 작업 첫 스크립트 머리말 기준, 추정)

| 바퀴 | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| C01 | 5개국 커버리지·RTA 민감도(WIPO 대조) | 원고 숫자 ↔ 결과·docx 대조 | N8 시험 레이블 독립 재추출 | 등록원부 기반 새 과제 후보(잠금 창) |
| C02 | NTIS·e-나라지표 대조 | 제공 KR 등록 반영률(권리목록 대조) | 3.1절 RTA 독립 재계산 | 등록원부 사건 범주·새 과제 |
| C03 | T1 최종 레이블 독립 재구성 | 수집본 재현·완결성 | 결과 감사 묶음(T3 동률·CN 결측, N6·N7·N9 매핑, NTIS 총액) | 등록원부·실시권 탐색 |
| C04 | 국세청 상태조회 편향(N4 모집단) | facts ↔ 원고 대조 | T3 층별 성적·출원인 집중 | 국민연금 스냅샷 점검 |
| C05 | IPC 개정(신설 코드) 대응 | 출원인 주소·후보 전체(N4·N7 선택 편향) | N9 레이블 독립 재구성 | NTIS 연결 차단·N9 겹침 |
| C06 | DART 원천 점검 | 원고 → docx·PDF 구조 대조 | N7 레이블 독립 재구성 | 벤처기업명단·새 과제 |
| C07 | KOSIS 작업 파일 ↔ 공식값, KSIC 9↔10 | 제출본 차이·N6 산술 | N8 재현 검증 | 권리 이전 사유 |
| C08 | 출원인 이름 사전·집계 | 원고 판별 차이 블록 | 등록원부 스키마·날짜 범위 | 새 레이블(원점 2012–2022) |
| C09 | 제공 범위 비율(권리목록 분모) | 원고 숫자 토큰 커버리지 | T3 시나리오 피처 재학습 | T4 납부기한 잡음·지연납부 피처 |
| C10 | 디자인·실용신안 자사 판정 재현(T1 R3) | 시드 변동·군집 부트스트랩 | 해외→KR 패밀리 연결 3종 | 원고 v10→v11 변경 감사 |
| C11 | 출원인코드↔법인 연결 타당성(체크섬·오연결·합병) | R9 TXT 배정 연결 팔 | US 번호 형식·오독 | 후보 단위 표(문서·출원인) |
| C12 | ETSI 원본 재파싱 | 원고 v11→v13 변경·옛값 잔존 | T3 출원인 집중 | CN·EP 개발 원점 단위 |
| C13 | 지식재산통계연보 심판 통계 | 트랙별 시드 묶음 분산(N4·N6–N9·T4) | ERRATA 반영 상태 | 새 과제 후보 목록 |
| C14 | 분산 추정·성분 부트스트랩 | v14↔v15 변경 감사·제출본 | T1 범위밖 피처 누락률 | 이원 표준오차 재현 |
| C15 | 연보 존속·등록 ↔ 등록원부 포착률 | T1·T3 시드 묶음 분산 | PKR-IPC 해외 배정 재생성 | 원고 판정 단위(주장 지도) |
| C16 | 이름·별칭 매칭 감사 | 시점 정합 전수 지도 | CNIPA 공식 통계 대조 | 변경 감사·ERRATA 반영 |
| C17 | 트랙 R3 모듈로 재검증(쓰기 가드) | 레이블 근거 문서 판정 | 시점 정합 지도 v2·시험 빌드 | 제공 US 메타 구조 |
| C18 | 연보 월별 등록·포착률 | split34·map5 재계산(US 이중 수록) | ERRATA 상태표·facts·빌드 | 후보별 평가 프레임 |
| C19 | JP·EP 포착률 | T4 패밀리 수 변형 | split34·map5·라벨 재계산 | ERRATA 등급 표본 감사 |
| C20 | 형식 확인만(`x0_peek.py`) | 형식 확인만(`e0_peek.py`) | facts 패치 ↔ 원천 토큰 대조 | 없음 |

C20은 작업 수가 적고 반박 폴더가 없어 중간에 멈춘 바퀴로 보인다(추정).

## 7. `분석/보고서_20261012/` — 빌드·그림·숫자 고정표

원고(`manuscript_*.md`), 그림 PNG, 숫자 고정표 CSV, docx·PDF는 저장소에 없다. 순서는 숫자 고정표 → 그림 → docx → PDF·doc이다.

### 7.1 숫자 고정표 `facts/build_facts.py`

| 항목 | 내용 |
|---|---|
| 실행 | `facts/` 폴더에서 `python -X utf8 -I build_facts.py`(머리말) |
| 입력 | `<MC>` 아래 결과 파일(읽기 전용): 공식 9개 트랙 `final_result*.json`, 필수성 `summary_dev.csv`·`essential_table_pub.csv`, R8–R12 결과(`NONCIRC_R8/res/*`, `TXT_R9/res/*`, `_shared_report/*`, `_ideas_R7·R8/*.json`), `<EXT>/firm_nps_link_rates.csv`·`ntis_link_rates.csv`. 몇 개 건수는 `registry_full.sqlite`·`business_rights.sqlite`를 `mode=ro`로 직접 센다 |
| 출력 | 같은 폴더 `facts.csv`(id, 주제, 수치, 단위, 문장 조각, 출처 경로, 조건·경고, note), `facts_mismatch.csv`(문서 값과 다른 항목) |
| 의존성 | 표준 라이브러리만(csv, json, sqlite3, decimal) |

### 7.2 그림 `figs/*.py`

| 파일 | 역할 |
|---|---|
| `_style.py` | 공통 스타일 v3: 폭 17cm(6.69in) 고정, 높이는 그림마다 cm 지정, 300dpi, bbox 자르기 없음, 그림 안 최소 글자 7pt, 맑은 고딕. `save()`가 저장 전에 작은 글자·잘림·글자 겹침·상자 넘침을 검사해 출력하고, `csv()`가 그림 데이터를 같은 이름 CSV(utf-8-sig)로 남긴다. 데이터 경로는 `figs/../../모델경쟁_20260927`(상대경로) |
| `make_diagrams.py` | 도식 `fig01_framework`(데이터 연계 프레임워크), `fig10_validation_design`(검증 설계). 데이터 입력 없음 |
| `make_data_figs.py` | 데이터 그림. 인자 없이 실행하면 원고에 쓰는 6개(`fig02`, `fig03`, `fig04`, `fig05`, `fig08`, `fig09`)를 그린다. `python make_data_figs.py fig03 fig05`처럼 이름을 주면 그것만. `fig06`·`fig07`은 함수만 있고 목록(`USED`)에 없어 명령줄로는 그려지지 않는다 |

데이터 그림의 입력(모두 `<MC>` 아래, 모듈을 불러올 때 KSIC 이름표 `map5/m2_ksic_profile.csv`를 먼저 읽는다):

| 그림 | 입력 |
|---|---|
| fig02 9개 트랙 성적표 | 코드 안에 적은 값(출처 표기 'R6_결과_20261004.md §3.1·§3.2') |
| fig03 필수성(기업 업종 일치) | `_shared_report/label_origin_R10/confirm_R12/res_main.csv`, `_shared_report/label_origin_R10/res_p1.csv` |
| fig04 코드 안 분화 | `_shared_report/fusion_advantage/split34/t5_cases.csv`, `fig_smallmultiples_long.csv` |
| fig05 RTA 지형도 | `_shared_report/fusion_advantage/map5/fig1_rta_map_P3.csv` |
| fig06(원고 미사용) | `map5/fig2_quadrant_tech_export.csv` |
| fig07(원고 미사용) | `T3_기술셀부상예측/R3/future_forecast_r3_ksic.csv` |
| fig08 T1 대기실 지수 | `T1_산업대기실/R3/future_forecast_r3.csv` |
| fig09 N9 상위 k | `N9_정부RnD과제_특허성과예측/final_result_r6.json` |

출력은 `figs/` 폴더의 `figNN_*.png`와 같은 이름 `.csv`다. 실행은 `figs/` 폴더에서(또는 경로를 주고) `python -X utf8 make_diagrams.py`, `python -X utf8 make_data_figs.py`. 이웃 모듈 `_style.py`를 import하므로 `-I`는 쓰지 않는다. 필요한 것: matplotlib, pandas, numpy, 맑은 고딕 글꼴.

### 7.3 빌드 `build/`

| 파일 | 사용법 | 하는 일 |
|---|---|---|
| `md2docx.js` | `node md2docx.js <원고.md> <그림 폴더> <출력.docx>` | 마크다운 원고를 공모전 제출용 docx로 바꾼다. `# [부록` 또는 `# [편집 메모` 제목 앞까지만 변환. A4·여백 20mm, 맑은 고딕 10.5pt(표 8.5pt, 캡션 9.5pt), 쪽 번호 바닥글, 첫 `---`는 쪽 나눔. 표 머리글 음영, 표 행은 쪽 경계에서 나뉘지 않게(`cantSplit`, main 5c94f0a에서 추가). 그림 자리표시 줄 `[그림 n: 파일.png] 캡션`을 그림으로 바꾸고, 파일이 없으면 빨간 '[그림 파일 없음]'을 넣는다. `**굵게**`, `*기울임*`, 글머리표·번호 목록, `**표 n.**`·`그림 n.` 캡션, `주:`·`출처:` 주석을 처리 |
| (환경변수) | `FIG_NATIVE=1` | 그림을 그린 물리 크기(300dpi 가정) 그대로, 본문 폭 642px를 넘으면 비율 유지 축소. 없으면 폭 620px·최대 높이 `FIG_MAXH`(기본 430px) |
| `render.ps1` | `powershell -NoProfile -ExecutionPolicy Bypass -File render.ps1 "<docx 절대경로>"` | MS Word로 같은 이름 PDF를 만들고 `PDF:`·`PAGES:`(쪽수)를 출력. 머리말의 '쪽 이미지(PyMuPDF)' 단계는 이 스크립트 안에 없고, 워크플로에서 PyMuPDF로 따로 했다(r13.js, final_polish.js) |
| `save_doc.ps1` | `powershell -File save_doc.ps1 "<docx 절대경로>"` | MS Word로 Word 97-2003 `.doc`를 만들고 `DOC:`·`PAGES:` 출력(공고문 제출 형식 대응, main 5c94f0a에서 추가) |

팀 PC에서 쓴 명령(워크플로 r13.js·final_polish.js에 적힌 그대로, bash 기준):

```bash
cd "<ROOT>\분석\보고서_20261012\build"
export NODE_PATH="$(npm root -g)"          # 전역 설치한 docx를 찾게 한다
FIG_NATIVE=1 node md2docx.js ../manuscript_v2_from_response.md ../figs ../보고서_초안_v3.docx
powershell -NoProfile -ExecutionPolicy Bypass -File render.ps1 "<보고서_초안_v3.docx 절대경로>"
```

PowerShell에서는 `$env:NODE_PATH = (npm root -g); $env:FIG_NATIVE = '1'; node md2docx.js …`. Word COM은 절대경로를 요구한다. 검증 루프의 `build_test/` 사본 중 C12–C16은 이전 판, C17–C19는 5c94f0a 직전 판과 같다(`cantSplit` 없음).

## 8. 초기 단계 코드

| 위치 | 이름 패턴 | 내용(머리말 기준) |
|---|---|---|
| `분석/*.py` (95) | `pilot_padi.py`, `padi_v2.py`, `padi_v3_family.py`, `model_*.py`, `predictive_validity.py`, `decompose_signal.py` | PADI 지표 설계·모형 탐색 |
| | `adv_*.py`, `defense_*.py`, `stress_test*.py`, `split_sensitivity.py`, `window_sensitivity.py`, `verify_df_vs_collinearity.py` | 적대적 검증, 심사 반론 방어 검정, 민감도 |
| | `build_*.py`, `test_*.py`, `leadlag_*.py`, `fetch_hs6_*.py` | 패널·통제변수 구축, 선행성 검정, Comtrade 수집 |
| | `check_*.py`, `diag_*.py`, `inspect_*.py`, `probe_*.py`, `scope_biz.py`, `verify_*.py`, `audit_registry.py`, `final_check*.py` | 수집 데이터 점검. 일부(`diag_biz_*.py`, `diag_rc001.py`, `probe_*.py`)는 저장소 밖 `수집파이프라인/collect_registry.py`를 import한다 |
| | `validate_*.py`, `fusion_*.py`, `verify_fusion.py`, `figures_v2.py`·`figures_v3.py` | 융합데이터 라벨 타당성, 고유 자산, 그림 |
| `분석/대기실_파일럿/` (7) | `pilot_clf.py`, `pilot_entry*.py`, `adv_entry*.py`, `common.py` | T1 산업 대기실의 전신 파일럿(추정). `공모전데이터/...` 상대경로와 세션 임시 폴더 경로를 쓴다 |
| `분석/fusion/` (2) | `synth_final_scope*.py` | 4개 진단 각도를 통합한 최종 사양 재계산 |

## 9. 추정·확인 못 한 점

**추정으로 남긴 부분**

1. T2·T5·T6, N1–N3이 공식 성적표에 없는 이유(탈락·보류). 코드에는 결론이 없다.
2. N3의 출처를 `_ideas_R2/global`로 본 것(머리말이 같다는 점만 근거).
3. 트랙·라운드 안의 실행 순서(→). 머리말·파일 번호·입출력 관계로 짐작했고, 실행 기록으로 확인하지 않았다.
4. `_workflows/*.js`를 돌린 실행기(Claude Code 워크플로 기능). 코드에는 전역 함수만 있고 실행기 이름이 없다. 2–5라운드에 워크플로 파일이 없는 이유도 추정이다.
5. 검증 루프 바퀴별 작업 요지(6.4절)는 각 작업 첫 스크립트 머리말에서 뽑았다. 번호 없는 `verify/` 폴더의 성격, C20이 중단된 바퀴라는 해석도 추정이다.
6. `작업데이터/*`(`kr_patents.parquet` 등)를 만든 코드의 위치. 저장소에서 찾지 못했다.
7. `<EXT>`의 일부 파일(`firm_nps_wkpl_*`, `R2global_*` 등)을 어느 스크립트가 썼는지는 파일 이름과 소비 코드 경로로 짐작했다.
8. `_shared_report/build_ntis_*.py`의 작성 시기(R4–R10). 입력에 R4 산출물이 있고, r11.js가 결과 `ntis_tables.md`를 입력으로 읽는다는 점만 근거다.

**확인하지 못한 점**

1. 어떤 코드도 실행하지 않았다. 각 스크립트가 지금 팀 PC 자료로 끝까지 도는지는 모른다.
2. 동결 해시 대조(N5 `lib_sha256`, N6·N9 `code_sha256`)가 이 저장소에서 체크아웃한 파일로도 통과하는지. 저장소는 `.gitattributes`로 줄바꿈을 LF로 정규화하므로, 팀 PC 원본 파일이 CRLF였다면 해시가 다르다(명세 json이 저장소에 없어 대조 불가).
3. `requirements.txt`의 버전이 분석 당시 쓴 판과 같은지, npm `docx` 9.7.1 외 판에서 `md2docx.js`가 도는지.
4. 머리말이 언급하는 `run_final_r3.sh`(T1 R3) 같은 셸 스크립트, `numcheck.py`(final_polish), 규칙 문서(`PROTOCOL_R*.md` 등)는 저장소에 없어 내용을 보지 못했다.
5. 출력 파일 이름은 코드의 경로 문자열에서 읽었다. 변수로 조합되는 이름(`f'{...}'`)은 일부만 적었다.

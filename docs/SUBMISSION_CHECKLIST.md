# 코드 제출 전 점검표

분석 코드를 제출(또는 외부 공유)하기 전에 위에서부터 차례로 확인한다. 구조 설명은 [CODE_GUIDE.md](CODE_GUIDE.md)에 있다.

- 기준: `main` 커밋 `5c94f0a` + 이 점검표를 추가한 `cloud/code-guide` 브랜치. 2026-11-02에 저장소 사본에서 검사한 결과를 함께 적었다.
- 자동 검사 도구: `python tools/make_code_package.py --check-only` (커밋 가드 `.githooks/guard.py` 규칙 + 이메일·절대경로 검사).

## A. 넣으면 안 되는 것

공모전 서약서 6항(제공 데이터는 팀원 외 공유 금지, 종료 후 폐기)과 README 규칙에 따른다.

- [ ] **제공 데이터·파생 데이터가 없다.** 데이터·압축 확장자(`.parquet` `.sqlite` `.db` `.npy` `.npz` `.pkl` `.feather` `.h5` `.zip` `.gz` `.7z` `.tar` `.xlsx` `.xls` `.tsv` 등)와 허용 폴더 밖 `.csv`·`.json`이 추적 파일에 없어야 한다.
  - 확인: `python tools/make_code_package.py --check-only` → '커밋 가드 통과'. (2026-11-02: 통과, 추적 파일은 `.py`·`.js`·`.ps1`과 루트 설정 파일뿐)
- [ ] **5MB를 넘는 파일이 없다.** 같은 검사에 포함. (2026-11-02: 가장 큰 파일은 `분석/보고서_20261012/facts/build_facts.py` 약 91KB)
- [ ] **결과표·그림·원고가 없다.** `.csv` `.json` `.png` `.docx` `.pdf`, 원고·결과 문서 `.md`(README·`docs/` 제외), 원장 `.txt`가 없어야 한다. `.gitignore`가 기본으로 막는다.
- [ ] **API 키가 없다.**
  - 키 원장 `API키_기록.txt`는 저장소 루트에 두고 커밋하지 않는다(guard가 이름으로 차단).
  - 키 문자열 대조는 guard가 **저장소 루트**에서 원장을 찾을 때만 한다. git worktree나 새 클론에서는 '키 패턴 0개 대조'가 되고 하드코딩 키 패턴만 검사된다. 최종 검사는 원장이 있는 **팀 PC 원본 작업 폴더**(`<ROOT>`)에서 `--strict`로 한다.
  - 키를 담은 수집 코드 `수집파이프라인/collect_registry.py`, `collect_kosis_new.py`는 저장소 밖에 있다(guard가 이름으로 차단). 저장소의 일부 초기 스크립트(`분석/diag_biz_*.py`, `diag_rc001.py`, `probe_*.py`)가 `collect_registry`를 import하므로, 이 스크립트들은 팀 PC에서만 돈다.
- [ ] **개인정보가 없다.**
  - ⚠ 2026-11-02 검사에서 **개인 이메일 주소가 하드코딩된 파일 3개**를 찾았다(OpenAlex polite pool의 연락처 `mailto` 값). 제출 전에 지울지(예: 환경변수로 받기) 팀이 정한다. 이 점검표 작업에서는 기존 코드를 고치지 않았다.
    - `분석/모델경쟁_20260927/_shared/ext/r3oa_work/r3oa_04_fetch.py` (11행)
    - `분석/모델경쟁_20260927/_shared/ext/r3oa_work/r3oa_07_fetch2.py` (14행)
    - `분석/모델경쟁_20260927/_shared/ext/r3ros_04_oafield.py` (42행)
  - 주민등록번호·휴대전화 번호 형식은 0건이었다(정규식 검사).
  - 코드 주석·상수에 제공 데이터 레코드(출원번호·발명 명칭·출원인 이름 예시)가 박혀 있는지는 자동 검사가 없다. 제출 범위의 파일을 눈으로 훑는다.
  - git 커밋 작성자 정보(이름·이메일)는 zip(`git archive`)에는 들어가지 않는다(커밋 ID만 zip 주석에 들어감). 저장소 자체를 공유하면 커밋 이력으로 보인다.
- [ ] `--strict` 검사가 통과한다: `python tools/make_code_package.py --strict --check-only` (원장 없음·이메일 후보 경고도 실패로 처리). 위 이메일 3건을 그대로 두기로 했다면 `--strict` 없이 만들고, 그 결정을 제출 기록에 남긴다.

## B. 로컬 경로 의존(제출물에 명시할 것)

이 코드는 팀 PC의 폴더 배치와 로컬 데이터를 전제로 한다. 받는 쪽이 클론만으로 실행할 수 없다는 점을 제출 설명에 적는다.

- [ ] **절대경로**: 코드 1,026개 파일이 프로젝트 루트 `C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전`을 절대경로로 쓴다(그중 검증 루프 763개, 모델경쟁 171개). 다른 PC에서는 같은 경로에 폴더를 두는 편이 낫다. 동결 코드의 경로를 고치면 N5·N6·N9 실행기의 SHA-256 대조(`lib_sha256`, `code_sha256`)가 깨진다.
- [ ] **그 밖의 사용자 폴더 경로 5개 파일**: 에이전트 작업 세션의 임시 폴더를 쓰는 `분석/대기실_파일럿/pilot_clf.py`·`pilot_entry.py`·`pilot_entry_sub.py`, `_shared/ext/firm_nps_fetch.py`(환경변수 `NPS_RAW`가 없을 때의 기본값), 전역 npm 폴더를 쓰는 `검증루프_20261012/C19/C19-T3/scripts/s14_build.py`. 다른 PC에서는 존재하지 않는 경로다.
- [ ] **상대경로**: T3 1라운드 `build_base.py`(`../_shared/meta`), 보고서 그림 `figs/_style.py`(`../../모델경쟁_20260927`), `대기실_파일럿/*.py`(`공모전데이터/...`)는 폴더 배치를 그대로 유지해야 한다.
- [ ] **저장소에 없는 입력**: 제공 융합데이터(`공모전데이터/융합데이터_*`), 메타 parquet(`_shared/meta/`), 작업데이터, 외부데이터, 수집 DB(`수집파이프라인/*.sqlite`), 외부데이터 수집본(`_shared/ext/*.parquet|csv`), 동결 명세(`final_spec*.json`), 캐시·결과. 목록은 CODE_GUIDE 2.1절.
- [ ] **외부 프로그램**: Windows, MS Word(`render.ps1`·`save_doc.ps1`의 COM 자동화), Node.js + npm `docx`(팀 PC 전역 9.7.1, `NODE_PATH` 필요), 맑은 고딕 글꼴, Python 3.13(팀 PC 3.13.2) + `requirements.txt`.
- [ ] **네트워크·키가 필요한 코드**: `_shared/ext`의 수집 스크립트(`*_collect*.py`, `*_fetch*.py`), N2 `n01_dart_collect.py`, N1 `c02_fetch_final_exports.py`, `_ideas_R2`의 수출 수집, `_ideas_R7/d*_collect*.py` 등. 수집은 이미 끝났다. 다시 돌리면 외부 API를 호출하므로 돌리지 않는다.

## C. 실행 순서 요약(팀 PC, 재현이 꼭 필요할 때만)

데이터가 있는 팀 PC에서만 의미가 있다. 최종시험(`--mode final`)은 '사전 고정 후 1회' 규칙으로 이미 끝났으므로 다시 돌리지 않는다.

1. **환경**: `pip install -r requirements.txt`(첫 줄 coding 표시 덕분에 한국어 Windows에서도 읽힌다), `npm install -g docx`.
2. **제공 데이터 → 메타**: `python 분석/모델경쟁_20260927/_shared/build_meta.py` → `_shared/meta/*.parquet`.
3. **외부데이터 가공**(수집본이 있을 때): `_shared/ext`의 `*_build*.py` → 트랙이 읽는 parquet. 수집 스크립트는 돌리지 않는다.
4. **공식 9개 트랙 개발 원점 재현**(`--mode dev`, 결과를 동결값과 대조):

   | 트랙 | 명령(트랙 폴더 기준) |
   |---|---|
   | T1 | `python R3/final_run_r3.py --mode dev` (트랙 폴더에서) |
   | T3 | `python R3/final_run_r3.py --mode dev` |
   | T4 | `python R2/final_run_r2.py --mode dev` |
   | N4 | `python final_run_r4.py --mode dev` |
   | N5 | `python final_run_r4.py --mode dev` |
   | N6 | `python final_run_r5.py --mode dev` |
   | N7 | `python final_run_r5.py --mode dev` |
   | N8 | `python final_run_r6.py --mode dev --part 1` → `--part 2` |
   | N9 | `python final_run_r6.py --mode dev --part 1` → `--part 2` |

   9분 시간 예산에 걸려 종료코드 3으로 끝나면 같은 명령을 다시 실행한다(캐시에서 이어감). 각 트랙의 준비 단계·실험·검증 스크립트 순서는 CODE_GUIDE 3절.
5. **필수성·공동 분석**(R7–R12): 각 트랙 `ESSENTIAL_R7/`, `NONCIRC_R8/`, `TXT_R9/`, `_shared_report/` — 순서는 CODE_GUIDE 3절 표.
6. **보고서**: `facts/build_facts.py`(숫자 고정표) → `figs/make_diagrams.py`, `figs/make_data_figs.py`(그림) → `build/md2docx.js`(`FIG_NATIVE=1`, docx) → `build/render.ps1`(PDF·쪽수) → `build/save_doc.ps1`(.doc). 명령은 CODE_GUIDE 7절.
7. **코드 패키지**: D절.

## D. 패키지 만들기와 확인

- [ ] 작업 트리가 깨끗하다(`git status`에 바뀐 추적 파일이 없다). 도구는 작업 트리와 묶을 커밋이 다르면 멈춘다(종료코드 2).
- [ ] 묶을 커밋을 정했다(기본 `HEAD`, 다른 커밋은 `--ref`).
- [ ] 원장이 있는 팀 PC 원본 작업 폴더에서 실행한다:
  ```bash
  python tools/make_code_package.py --strict
  # → <저장소>/dist/분석코드_패키지.zip (zip 안 최상위 폴더 '분석코드_패키지/')
  ```
  출력된 파일 수와 SHA-256을 제출 기록에 남긴다. zip 주석에는 묶은 커밋 ID가 들어간다(`git archive` 기본 동작).
- [ ] 도구가 하는 일: (1) `git ls-tree`로 추적 파일 목록 (2) 작업 트리 = 커밋 확인 (3) 목록 전체에 `guard.py` 재실행(막히면 zip을 만들지 않고 종료코드 1) (4) 이메일·절대경로 검사 (5) `git archive --format=zip` (6) zip 목록이 검사한 목록과 같은지, zip 무결성 확인.
- [ ] zip을 풀어 목록을 눈으로 확인한다: `분석/` 코드, `README.md`, `docs/`, `requirements.txt`, `tools/`, `.githooks/`, `.gitignore`, `.gitattributes`만 있어야 한다.
- [ ] zip은 커밋하지 않는다(`dist/`와 `*.zip`은 `.gitignore`와 guard가 막는다).
- [ ] 공고문이 코드 제출 형식(파일 형식·크기·폴더 구성)을 따로 정했다면 그에 맞춘다(이 점검표 작성 때 확인하지 못했다).

## E. 저장소 관리 메모

- `.gitignore`는 '전부 제외 후 `분석/` 아래 코드만 허용' 방식이라 `docs/`, `tools/`, `requirements.txt`는 처음 추가할 때 `git add -f`로 넣었다. 이미 추적 중인 이 파일들의 수정은 평소처럼 커밋된다. 같은 폴더에 새 파일을 더할 때도 `-f`가 필요하다(또는 팀이 `.gitignore`에 예외를 추가).
- 커밋 가드는 `git config core.hooksPath .githooks`로 켠다. 커밋 전에 `python .githooks/guard.py <파일…>`로 직접 돌려 볼 수 있다.

# C18-T3 s6 생성기: C16-T4 s3_facts.py(102행 통합 패치 절차)를 읽어 v19·C18 경로로 바꾸고, C16·C17 facts 항목과 v18·v19 부록 변경 동기를 덧붙인 s6_facts.py를 만든다.
# 원 스크립트는 읽기만 한다. 사용: python -I s6_make_facts_script.py && python -I s6_facts.py
import sys, hashlib
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
SRC = AN + r"\검증루프_20261012\C16\C16-T4\scripts\s3_facts.py"
DST = AN + r"\검증루프_20261012\C18\C18-T3\scripts\s6_facts.py"
s = open(SRC, encoding="utf-8").read()
print("C16 s3_facts.py sha1", hashlib.sha1(s.encode("utf-8")).hexdigest()[:8])
rep = [
 ('# C16-T4 s3: facts 단일 패치', '# C18-T3 s6(C16-T4 s3_facts.py에서 생성): facts 단일 패치'),
 ('현재 원고(4cbf3d55)', '현재 원고(v19 b316df8e)'),
 ('OUT = L + r"\\C16\\C16-T4"; WK = OUT + r"\\facts_work"', 'OUT = L + r"\\C18\\C18-T3"; WK = OUT + r"\\facts_work"'),
 ("'원고 부록 표기(v17 %d행): %s'", "'원고 부록 표기(v19 %d행): %s'"),
 ("C16 = 'C16(부록 반영분 동기)'", "C16 = 'C16(부록 반영분 동기)'; C18 = 'C18(C16·C17 facts·v18–v19 부록 동기)'"),
 ('open(OUT + r"\\facts_patch_consolidated.csv"', 'open(OUT + r"\\facts_patch_v19.csv"'),
 ('open(OUT + r"\\s3_out.json"', 'open(OUT + r"\\cache\\s6_out.json"'),
]
for a, b in rep:
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
EXTRA = r'''
# ==== C18 추가: v18·v19 부록 변경 동기 + C16·C17 facts 항목(PREREG 6절) ====
ms_sha = hashlib.sha1(open(R + r"\manuscript_v3.md", "rb").read()).hexdigest()
assert ms_sha.startswith("b316df8e"), ms_sha
# (1) E-C16T3-F + v18 부록 변경(LINK-NTIS·C-COUNTRY-SHARE) 동기
op_set(FS, 'DATA-KR-APP', '정확한 문장 조각', '제공 KR 출원 1,087,511건(특허 1,036,864·실용신안 50,647)', 'E-C16T3-F; C16-T3 kr_prefix.csv(C18 재계산 일치)', C18)
op_set(FS, 'LINK-NTIS', '정확한 문장 조각', app('LINK-NTIS'), 'E-C16T3-01 반영(부록 %d행 v19) 동기' % appl('LINK-NTIS'), C18)
op_app(FS, 'LINK-NTIS', 'note', '분모·분자 실용신안 포함(실용신안 연결 83건); 특허만 9.78%·등록 11.67%(C16-T3)', 'E-C16T3-F', C18)
op_app(FS, 'LBL-CW-5D', 'note', '분모 KR 출원 1,087,511은 실용신안 포함(C16-T3)', 'E-C16T3-F(선택)', C18)
op_set(FS, 'C-COUNTRY-SHARE', '정확한 문장 조각', app('C-COUNTRY-SHARE'), 'E-C16T3-02 반영(부록 %d행 v19) 동기' % appl('C-COUNTRY-SHARE'), C18)
# (2) E-C16T1-F1: C16-T1-VIS(부록 535행 값을 문장 조각으로, 4자리 값은 조건 열)
op_add(SP, S('C16-T1-VIS', '3.3.1, 4.4', app('C16-T1-VIS'), r'분석\검증루프_20261012\C16\C16-T1\variants.csv',
             cond='원점 가시 모집단 S1-full·L-a 0.4607 [0.4423, 0.4767](독립 −0.0112, 짝지은 Δ CI [−0.0192, −0.0054]), S1-fixed 0.4671(짝지은 CI 0 포함), S2 0.4713, L-b 0.5453; 의존 쌍 7.30%; M0·SBB 분류 불변. S1만 고친 판이며 원점 정보 전체 결합판은 C17-T1-CMB(본문 410행 0.464–0.471)'),
       'E-C16T1-F1 + E-C17T1-F1 비고 권고; 부록 %d행 v19' % appl('C16-T1-VIS'), C18)
# (3) E-C17T1-F1: C17-T1-NEWC(부록 행 있음)·RC·CMB
op_add(SP, S('C17-T1-NEWC', '3.3.1, 4.4', app('C17-T1-NEWC'), r'분석\검증루프_20261012\C17\C17-T1\variants.csv, verify_0\v0_out.json, verify_1\v1_out.json',
             cond='T1-02 신규진입자 1,304법인 모두 범위안 첫 등록 2020–25(원점 이후); 원점에 날짜 확인되는 범위안 기록 232곳(주 정의 하한; 비모호 owner 매칭 최대 245, 범위밖 등록 포함 1,197); N1(재계산) skill 0.4781 [0.4601, 0.4962]; N2 +0.0064; N3 −0.0006~−0.0053(공여 점수, 범위 보고용); 무작위 동수 제외 +0.0052로 같음(구성 효과)'),
       'E-C17T1-F1(verify 반영 수정: 232곳 정의·무작위 동수); 부록 %d행 v19' % appl('C17-T1-NEWC'), C18)
op_add(SP, S('C17-T1-RC', '3.3.1, 4.4', 'T1-05 RC:001 법인 308곳(위험쌍 14,801·양성 6): R1 재계산 Δ +0.0056 [+0.0037, +0.0081], 고정 +0.0001, 경계 [−0.052, +0.0087]',
             r'분석\검증루프_20261012\C17\C17-T1\variants.csv',
             cond='경계는 기준선 재구축 값 고정 시(기준선도 극단이면 [−0.0553, +0.0356]); Δ는 기준선 약화 몫(RC 행 자체 약 +0.00004); RC 행 무작위 대입 +0.0005~+0.0019'),
       'E-C17T1-F1(verify 반영 수정)', C18)
op_add(SP, S('C17-T1-CMB', '3.3.1, 4.4', '원점 정보 결합(S1-full+N1+R1) skill 0.4711 [0.4518, 0.4909] Δ −0.0016 [−0.0110, +0.0087](고정 0.4696); S1+N1 0.4642(재계산)·0.4689(고정); 원점 정보판 범위 0.464–0.471',
             r'분석\검증루프_20261012\C17\C17-T1\variants.csv', cond='본문 410행 0.464–0.471의 원천(C18 재계산 일치); 모두 영향 없음, M0·SBB 분류 유지'),
       'E-C17T1-F1', C18)
# (4) E-C17T2-F1(E-C16T1-F2 대체) + E-C17T2-F2
op_add(SP, S('C17-T2-ALIAS', '3.3.2, 4.4', app('C17-T2-ALIAS'), r'분석\검증루프_20261012\C17\C17-T2\sens_results.csv, cache\s2_out.json',
             cond='이름 경로 전용 양성 2,990쌍: 코드 판정 1,001(권리목록 979 오매칭 0, 등록원부 22 중 오매칭 20), 주소 플래그 208쌍 중 오매칭 추정 112; 오매칭률 E2 5.0% [4.0, 6.3](V7 4.1%, V8a 4.2%, E1 4.9%, 위약 보정 4.4%); Δskill(제외) +0.0066 [0.0041, 0.0087], (음성 전환) +0.0065, 하한 +0.0002, V9 +0.0087, 유리 결합 +0.0036; R 코드 확정 오매칭 121쌍(같은 계보 의심 20쌍), 되돌림 Δ(iii) −0.0001~+0.0011(양의 값은 공개 전용 양성 16쌍 처리 때문, 그중 5쌍은 진입 문서도 다른 코드 출원), 확률판 −0.0001~+0.0013, 결합 Δ(iv) +0.0067~+0.0077; kappa 3범주 0.987(판정 가능 134쌍 0.853, 오매칭 n=4). C16-T1-ALIAS 표본 320 수치(+0.0112/+0.0014/+0.0125)는 C16 표본 수치로만 남기고 본문 근거로 쓰지 않음(E-C17T2-F2). 상한 6.3% > 5%라 \'정합\'으로 쓰지 않음'),
       'E-C17T2-F1(수정판: R 보조·kappa) + E-C17T2-F2; 부록 %d행 v19' % appl('C17-T2-ALIAS'), C18)
# (5) E-C17T3-F1: C17-T3-NEWIPC(부록 행 있음)·C17-NTIS-COUNT·C17-PIT-MAP-V2(조건부). '혼합' 해석 문장은 verify 반박이라 뺌.
op_add(SP, S('C17-T3-NEWIPC', '3.2.2, 4.4', app('C17-T3-NEWIPC'), r'분석\검증루프_20261012\C17\C17-T3\t3_newipc_sens.csv; cache\t3_sens_out.json',
             cond='IPC 2021.01에 없고 이후 판에만 있는 메인그룹 90셀(하위클래스 신설 72·메인그룹 신설 18; H10*·B64U 48셀 포함): 제외(레이블 유지) 0.406 [0.349, 0.463], 제외+레이블 재선정 0.380 [0.321, 0.438], 구 IPC 되돌림 0.400 [0.341, 0.458]; 개발 원점 평균 Δ −0.006/−0.005/−0.008. 동결 예측 재채점(재학습 없음). 본문 409행 0.38–0.42 하한의 원천(H10만 빼거나 합친 C05-T1 0.391–0.416과 합침). 개발 대비 \'혼합\' 해석은 verify 반박으로 쓰지 않음(E-C17T3-D4)'),
       'E-C17T3-F1(facts_candidates.csv 1행, 반박된 혼합 문장 제거); 부록 %d행 v19' % appl('C17-T3-NEWIPC'), C18)
op_add(SP, S('C17-NTIS-COUNT', '3.4', 'NTIS 2026 스냅샷 과제 수 대 공식: 2018–21 1.00000, 2022 0.99996, 2023 0.99993, 2024 0.99995(원문 확인: KIST 2022·2023 PDF, 2024 보도); 금액 포착률 2010–24 0.985–1.005',
             r'분석\검증루프_20261012\C17\C17-T3\cache\ntis_count_check.csv; cache\ntis_amount_check.csv',
             cond='과제 수 공식값은 2018–24만, 2010–17은 금액으로 대체. 본문 257행 \'과제 수는 2018–24년 공식 통계와 같고\'의 원천(C18 재계산 0.99993–1.00000)'),
       'E-C17T3-F1(facts_candidates.csv 2행)', C18)
op_add(SP, S('C17-PIT-MAP-V2', '2.1, 2.5, 4.4(원고 숫자 기재 금지)', '시점 정합 지도 확정판: 트랙 86요소 A 16·B 11(B1 9, B2 2)·C 48(C* 12)·D 7·E 2·NA 2, 트랙 밖 2(C 1·NA 1); 트랙 B·C 59 중 v18 원고 공개됨 38·부분 21·미공개 0',
             r'분석\검증루프_20261012\C17\C17-T3\pit_map_v2.csv; grade_table.csv; grade_rule.txt',
             cond='조건부(정의 의존): 과제 한 줄 정의를 쓰면 B 33. 코더 kappa 86행 0.602·88행 0.596. 두 코더·재판정자는 같은 에이전트. 원고에 숫자로 쓰지 않음. E-C16T2-F1(C16-PIT-MAP) 대체'),
       'E-C17T3-F1(facts_candidates.csv 3행, 조건부)', C18)
# (6) v19 본문 409행 숫자 원천 보충(H10 몫 45%·범위 0.38–0.42)
op_add(SP, S('C18-T3-H10-RANGE', '4.4', 'H10 셀 45개가 부상 504셀의 4%이나 순증가의 45%(개발 원점 0–21%); H10만 빼거나 합쳐 재채점 0.391–0.416(C05-T1 S1–S3), 원점 이후 신설 90셀 제외+재선정 0.380(C17-T3) → 본문 0.38–0.42',
             r'분석\검증루프_20261012\C05\C05-T1\rescore_scenarios.csv; 분석\검증루프_20261012\C17\C17-T3\t3_newipc_sens.csv',
             cond='공식 0.423 불변; 0.38은 H10 외 신설 셀(G06V·B64U·메인그룹 신설)을 포함한 90셀 재선정판(문장 주어 H10과 대상 다름 — 선택 E-C17T3-01 권장판이 정확)'),
       'C18-T3 원천 재검(recheck_numbers.csv E-C17T3-01)', C18)
'''
mark = "# 쓰기(작업 사본)"
assert s.count(mark) == 1
s = s.replace(mark, EXTRA + "\n" + mark)
# 쓰기: 원본 BOM·CRLF 유지(PREREG 6절) — facts.csv·supplement는 BOM+CRLF, R12는 BOM 없음+CRLF
old_write = """def write(name, hdr, rows):
    with open(WK + '\\\\' + name, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, lineterminator='\\n'); w.writerow(hdr)"""
assert s.count(old_write) == 1, "write() 패턴"
new_write = """def write(name, hdr, rows):
    rawo = open(FD + '\\\\' + name, 'rb').read()
    bom = rawo.startswith(b'\\xef\\xbb\\xbf'); crlf = b'\\r\\n' in rawo
    with open(WK + '\\\\' + name, 'w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\\r\\n' if crlf else '\\n'); w.writerow(hdr)"""
s = s.replace(old_write, new_write)
s = s.replace("import csv, io, os, re, sys, json, shutil, hashlib", "import csv, io, os, re, sys, json, shutil, hashlib")
open(DST, "w", encoding="utf-8").write(s)
print("written", DST, len(s))

# C17-T3 s3: 시점 정합 지도 v2 — 코더 A(속성 → grade_rule 결정 나무), 코더 B(과제 한 줄 정의 문자 적용), kappa, 조정, 원고(v18) 공개 재판정, 집계.
# 입력: C16-T2 pit_map.csv(읽기만), t3 정량(cache\t3_sens_out.json), NTIS 재계산(cache\ntis_count_check.csv). 원고 sha1 확인.
import os, sys, json, hashlib
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, "검증루프_20261012", "C17", "C17-T3")
MS = os.path.join(ROOT, "보고서_20261012", "manuscript_v3.md")
assert hashlib.sha1(open(MS, 'rb').read()).hexdigest().startswith('39452163')
old = pd.read_csv(os.path.join(ROOT, "검증루프_20261012", "C16", "C16-T2", "pit_map.csv"), encoding='utf-8-sig')
assert len(old) == 88
T3 = json.load(open(os.path.join(W, 'cache', 't3_sens_out.json'), encoding='utf-8'))
sens = pd.read_csv(os.path.join(W, 't3_newipc_sens.csv'))
def sv(v, p): return float(sens[(sens.variant == v) & (sens.period == p)].skill.iloc[0] - (0.422887490953899 if p == 'final' else 0.4275039841486892))

# ---------------- 코더 A 속성: used, inc, dep, ev, lab, interp, unver, 근거 ----------------
# dep: none / event / taxonomy_new / vintage / snapshot ; ev: 비선택 근거(확인 등급 또는 이번 재계산)
A = {
 'T1-01': ('Y','Y','event','N','N','N','N','lib.py:59-63 cap=None, final_run_r3.py:75 — 2025-08까지 등록분으로 3건 계수'),
 'T1-02': ('Y','Y','event','N','N','N','N','prep1.py:18 applicant_query(융합데이터 등장 법인), DATA_NOTES:40'),
 'T1-03': ('Y','Y','event','N','N','N','N','lib.py:61-65,96 Mrisk==0(2025-08 등록 기준)'),
 'T1-04': ('Y','Y','event','N','N','N','N','final_run_r3.py:108,238 prior_any(2025-08까지 공개·등록)'),
 'T1-05': ('Y','Y','event','N','N','N','N','prep2.py:8 status OK만 — 2026 조회 실패(휴면 성향) 법인 제외'),
 'T1-06': ('Y','N','none','N','Y','N','N','r3.py:54-75,105-106 레이블 2025-08 등록·공개, 공란·지연 = 측정 한계'),
 'T1-07': ('Y','N','vintage','N','N','N','N','DATA_NOTES:16 별칭=현재 이름; 레이블 값 오귀속(양수 법인), 관측 가능 여부는 코드 경로가 보장'),
 'T1-08': ('Y','N','vintage','N','N','N','N','DATA_NOTES:11,15,41 2026 권리목록 현재 보유분 피처'),
 'T1-09': ('Y','N','vintage','N','N','N','N','final_run_r3.py:44-59 등록일≤cap, 보유 기준 2026'),
 'T1-10': ('Y','N','vintage','N','N','N','N','FINAL_REPORT_R3:32 ECOS 2020=100 현행 계열'),
 'T1-11': ('Y','N','vintage','N','N','N','N','r2ext.py io_link 34-35 2026-06 수정 접속표'),
 'T1-12': ('Y','N','snapshot','N','N','N','N','final_run_r3.py:89 기준선 패널 cap=None·DART 2026 업종(기준선 피처)'),
 'T1-13': ('Y','N','none','N','N','N','N','lib.py:84-93 엄격 모드 ≤원점 연말'),
 'T1-14': ('N','N','snapshot','N','N','N','N','final_run_r3.py:215-222 보조 변형만'),
 'T1-15': ('Y','Y','vintage','Y','N','N','N','위험집합·레이블이 제공 KSIC 라벨; 비선택 근거 F226(확인) 배정 단절 없음'),
 'T3-01': ('Y','Y','taxonomy_new','N','N','N','N','r3core.py:19 MIN_TOTAL=50; 셀 90개가 IPC 2021.01에 없는 메인그룹(t3_sens_out U_final)'),
 'T3-02': ('Y','N','vintage','N','N','N','N','r3lib.py:53-57 레이블 KR 공개 수(현행 분류); V2 되돌림 시 기존 셀 레이블 25개 변화'),
 'T3-03': ('Y','Y','taxonomy_new','N','N','N','N','V2 되돌림: 원점 시행판 코드 셀 6개가 적격이 되는데 공식 집합에 없음(문서가 신설 코드로 이동)'),
 'T3-04': ('Y','N','vintage','N','N','N','N','F116 C05-T1 피처 창 현행 분류(재학습 없이 재점수 불가)'),
 'T3-05': ('Y','Y','vintage','Y','N','N','N','셀 KSIC 축 = 제공 라벨; F226 단절 없음; 원점 2021은 KSIC 10차 시행 중'),
 'T3-06': ('Y','N','vintage','N','N','N','N','FINAL_REPORT_R3:44,88-91 KOSIS 10차 재작성'),
 'T3-07': ('Y','N','vintage','N','N','N','N','FINAL_REPORT_R3:45-49 RoS 2026-09판'),
 'T3-08': ('Y','N','none','N','N','N','N','FINAL_REPORT_R3:69-86 최초 공개 ≤t0'),
 'T3-09': ('Y','N','vintage','N','N','N','N','F199·F228·F231-236 출원인명 승계 소급'),
 'T3-10': ('Y','N','none','N','N','N','N','FINAL_REPORT_R3:61-67 공개연도 ≤t0'),
 'T4-01': ('Y','Y','none','N','N','N','N','t4lib.py:181-199 L 시점 납부·말소 cd≤L'),
 'T4-02': ('Y','Y','vintage','Y','N','N','N','모집단 50 KSIC 현행 라벨; F226'),
 'T4-03': ('Y','N','none','N','N','N','N','FINAL_REPORT_R2:18 납부기록 레이블, 관측 완료(캐시 2026-08)'),
 'T4-04': ('Y','N','none','N','Y','N','N','t4lib.py:216-233 학습 레이블 소멸 cd≤C, 회복 소급 0.1%'),
 'T4-05': ('Y','N','vintage','N','N','N','N','FINAL_REPORT_R2:83 서지 스냅샷'),
 'T4-06': ('Y','N','vintage','N','N','N','N','FINAL_REPORT_R2:10,65,83 KOSIS 최신 수정치'),
 'T4-07': ('Y','N','vintage','N','N','N','N','FINAL_REPORT_R2:10,66 family 현재값'),
 'T4-08': ('Y','N','vintage','N','N','N','N','t4lib.py:119,127,315-356 owner 현재 명칭'),
 'T4-09': ('Y','N','none','N','N','N','N','t4lib.py:236-246 날짜 ≤L'),
 'N4-01': ('Y','Y','event','N','N','N','N','n4lib.py:170-171 registered(2026 조회), 264 inner join; r3nts_01_targets.py:7-29 CA 2026'),
 'N4-02': ('Y','N','vintage','N','N','N','N','n4lib.py:199-205 출원→코드(현재 상호); 모집단은 조건2(원부 정확 코드)가 정함'),
 'N4-03': ('Y','Y','none','N','N','N','N','n4lib.py:265 end ≤ t0 제외(과거 날짜)'),
 'N4-04': ('Y','N','none','N','Y','N','N','DATA_NOTES:16-17 국세청 1회 조회 소급·지연 결손'),
 'N4-05': ('Y','Y','event','N','N','N','N','F88 C04-T1 학습 원점 생존 선별'),
 'N4-06': ('Y','N','vintage','N','N','N','N','DATA_NOTES:39 NPS 키 biz6+현재 상호'),
 'N4-07': ('Y','N','snapshot','N','N','N','N','DATA_NOTES:41 NTIS 2026 피처'),
 'N4-08': ('Y','N','snapshot','N','N','N','N','n4feat.py:239-247 DART 대상 목록(피처); E-C16T2-D2 반박: 상장 구성 효과'),
 'N4-09': ('Y','N','none','N','N','N','N','FINAL_REPORT_R4:39 시간여행 검사'),
 'N4-10': ('Y','Y','vintage','Y','N','N','N','모집단 조건2 제공범위 KR 등록특허(현행 라벨); F226'),
 'N5-01': ('Y','Y','none','N','N','N','N','n5lib.py:158-166 known≤t0말·refl≤t0말'),
 'N5-02': ('Y','N','none','N','Y','N','N','DATA_NOTES:10,23 ETSI 2026 추출, 창 끝 절단 3.2%'),
 'N5-03': ('Y','N','none','N','N','N','Y','DATA_NOTES:22 철회 이력 없음'),
 'N5-04': ('Y','N','vintage','N','N','N','N','n5lib.py:74-76 현재 청구항'),
 'N5-05': ('Y','N','vintage','N','N','N','N','n5lib.py:65,171 출원인명'),
 'N5-06': ('Y','N','snapshot','N','N','N','N','FINAL_REPORT_R4:15 NTIS 성과'),
 'N5-07': ('Y','N','vintage','N','N','N','N','DATA_NOTES:44 RoS'),
 'N5-08': ('Y','N','vintage','N','N','N','N','n5lib.py:98-103 family 현재값'),
 'N5-09': ('Y','N','none','N','N','N','N','FINAL_REPORT_R4:14 반영 ≤t0'),
 'N5-10': ('Y','Y','vintage','Y','N','N','N','n5lib.py:20 KSIC 26429 현행 라벨; F226'),
 'N6-01': ('Y','Y','snapshot','Y','N','N','N','n6_lib.py:133-135 NTIS 2026 수행기업; 비선택: 과제 수 2018–24 공식 0.99993–1.00000(이번 재계산, KIST PDF·보도 원문)'),
 'N6-02': ('Y','Y','event','N','N','N','N','n6_lib.py:143-157 미연결 기업 결과 구조적 0'),
 'N6-03': ('Y','N','none','N','Y','N','N','FINAL_REPORT_R5:115 2023-12 약 10% 미공개'),
 'N6-04': ('Y','N','vintage','N','Y','N','N','F132·F155 이름 경로 의존(등록번호 공란); kr_patents 출원인명 = 현재 권리자 이름(F395 C16-T1) → 레이블 값 빈티지(C), 측정 한계(D)는 부성격 [verify_0 수정]'),
 'N6-05': ('Y','N','snapshot','N','N','N','N','DATA_NOTES:47,52,60 NTIS 피처'),
 'N6-06': ('Y','N','vintage','N','N','N','N','DATA_NOTES:48 NPS 키'),
 'N6-07': ('Y','N','none','N','N','N','N','FINAL_REPORT_R5:41 ≤2010 사전'),
 'N6-08': ('Y','Y','vintage','Y','N','N','N','DATA_NOTES:11-12 50 KSIC 현행 라벨; F226'),
 'N7-01': ('Y','Y','event','N','N','N','N','n7lib.py:200-201 CA/권리목록 사업자번호 + 국세청 registered'),
 'N7-02': ('Y','Y','snapshot','Y','N','N','N','n7lib.py:330-331 NTIS 수행기관 제외; 비선택: 과제 수 2018–24 0.9999+, 금액 2010–24 0.985–1.005(이번 재계산)'),
 'N7-03': ('Y','N','none','N','Y','N','N','n7lib.py:334-335 NTIS 2026 첫 등장, 보안과제 관측 불가'),
 'N7-04': ('Y','N','vintage','N','N','N','N','DATA_NOTES:45 NPS 키'),
 'N7-05': ('Y','N','snapshot','N','N','N','N','DATA_NOTES:49-51 NTIS 성과·연구비'),
 'N7-06': ('Y','N','none','N','N','N','N','DATA_NOTES:47-48 날짜 ≤t0'),
 'N7-07': ('Y','N','vintage','N','N','N','N','n7lib.py:229 CA 현재 상호(피처); 모집단 조건은 원부 코드'),
 'N7-08': ('Y','Y','vintage','Y','N','N','N','DATA_NOTES:11,50 현행 KSIC; F226'),
 'N8-01': ('Y','Y','none','N','N','N','N','n8lib.py:41-66 봉인 추출, build_t0 205-230'),
 'N8-02': ('Y','Y','vintage','Y','N','N','N','DATA_NOTES:49 656,360 현행 라벨; F226'),
 'N8-03': ('Y','N','none','N','N','N','N','final_run_r6.py final_labels 예고등록 2023-24'),
 'N8-04': ('Y','N','vintage','N','N','N','N','n8lib.py:47,178 현재 청구항·IPC'),
 'N8-05': ('Y','N','vintage','N','N','N','N','n8lib.py:49-51 owner 현재 명칭'),
 'N8-06': ('Y','N','none','N','N','N','Y','DATA_NOTES:51 말소 실시권 재구성 불가'),
 'N8-07': ('Y','N','vintage','N','N','N','N','DATA_NOTES:13 family 현재값'),
 'N8-08': ('N','N','snapshot','N','N','N','N','final_spec_r6.json champion.feats 미포함'),
 'N8-09': ('Y','N','none','N','N','N','N','DATA_NOTES:12 봉인 추출 <2023-01-01'),
 'N9-01': ('Y','Y','snapshot','Y','N','N','N','DATA_NOTES:9,27 NTIS 2022 과제; 비선택: 2022 과제 수 76,049/76,052=0.99996(이번 재계산, KIST 2022 PDF)'),
 'N9-02': ('Y','N','none','N','Y','N','N','DATA_NOTES:10,28 성과 보고 2026-09·2023-12 미공개 = 측정 한계'),
 'N9-03': ('Y','N','snapshot','N','N','N','N','DATA_NOTES:28,41-44 성과 이력 2026 스냅샷'),
 'N9-04': ('Y','N','vintage','N','N','N','N','DATA_NOTES:55 과제 기재값 사후 수정'),
 'N9-05': ('Y','N','vintage','N','N','N','N','DATA_NOTES:48,56 기관명↔출원인명'),
 'N9-06': ('Y','N','none','N','N','N','N','DATA_NOTES:30,58 ≤2014 고정'),
 'N9-07': ('Y','Y','vintage','Y','N','N','N','DATA_NOTES:10 현행 KSIC; F226'),
 'X-01': ('Y','N','vintage','N','N','N','N','F278 C11-T2 상속 라벨(대체시험 3.6, 트랙 밖)'),
 'X-02': ('N','N','snapshot','N','N','N','N','F57 C02-T4 기각 과제만'),
}
assert set(A) == set(old['요소ID']), set(old['요소ID']) ^ set(A)
def tree(used, inc, dep, ev, lab, interp, unver):
    if used == 'N': return 'NA', ''
    if inc == 'Y' and dep == 'event': return 'B', 'B1'
    if inc == 'Y' and dep == 'taxonomy_new': return 'B', 'B2'
    if inc == 'Y' and dep in ('vintage', 'snapshot') and ev != 'Y': return 'B', 'B3'
    if dep in ('vintage', 'snapshot'): return 'C', ('C*' if inc == 'Y' else '')
    if lab == 'Y' or interp == 'Y': return 'D', ''
    if unver == 'Y': return 'E', ''
    return 'A', ''
ca = []
for k, v in A.items():
    g, sub = tree(*v[:7]); ca.append(dict(요소ID=k, used=v[0], inc=v[1], dep=v[2], ev=v[3], lab=v[4], interp=v[5], unver=v[6], 근거=v[7], 등급A=g, 하위=sub))
ca = pd.DataFrame(ca)

# ---------------- 코더 B: 과제 한 줄 정의를 요소 문구에 문자 그대로 (하위 규칙 없음) ----------------
B = {}
for k in A: B[k] = None
Bset = dict(
 B=['T1-01','T1-02','T1-03','T1-04','T1-05','T1-06','T1-07','T1-15','T3-01','T3-02','T3-03','T3-05','T4-02','T4-04','N4-01','N4-04','N4-05','N4-10',
    'N5-02','N5-10','N6-01','N6-02','N6-03','N6-04','N6-08','N7-01','N7-02','N7-03','N7-08','N8-02','N9-01','N9-02','N9-07'],
 C=['T1-08','T1-09','T1-10','T1-11','T1-12','T3-04','T3-06','T3-07','T3-09','T4-05','T4-06','T4-07','T4-08','N4-02','N4-06','N4-07','N4-08',
    'N5-04','N5-05','N5-06','N5-07','N5-08','N6-05','N6-06','N7-04','N7-05','N7-07','N8-04','N8-05','N8-07','N9-03','N9-04','N9-05','X-01'],
 A=['T1-13','T3-08','T3-10','T4-01','T4-03','T4-09','N4-03','N4-09','N5-01','N5-09','N6-07','N7-06','N8-01','N8-03','N8-09','N9-06'],
 E=['T1-14','N5-03','N8-06','N8-08','X-02'])
for g_, ks in Bset.items():
    for k in ks: assert B[k] is None, k; B[k] = g_
assert all(v is not None for v in B.values()), [k for k, v in B.items() if v is None]
cb = pd.DataFrame(dict(요소ID=list(B), 등급B=list(B.values())))
cb['비고'] = 'B 정의 문자 적용: 레이블 관측이 원점 이후 스냅샷·등록에 달리면 B, 보조 변형만은 E(기타)'

def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = np.mean(np.array(a) == np.array(b))
    pe = sum((np.sum(np.array(a) == c) / n) * (np.sum(np.array(b) == c) / n) for c in cats)
    return float((po - pe) / (1 - pe)), float(po)
mm = ca.merge(cb, on='요소ID')
k88, po88 = kappa(mm.등급A.tolist(), mm.등급B.tolist())
tr = mm[~mm.요소ID.str.startswith('X')]
k86, po86 = kappa(tr.등급A.tolist(), tr.등급B.tolist())
# 조정: 불일치 행 — 정의형/사실형
adj = []
for _, r in mm[mm.등급A != mm.등급B].iterrows():
    if r.등급A == 'D' and r.등급B == 'B': t, why = '정의형', 'grade_rule D(측정층): 결과창 관측 절단·지연·결손이며 관측 가능 여부가 단위의 원점 이후 연결에 달리지 않음 — ' + r.근거
    elif r.하위 == 'C*' and r.등급B == 'B': t, why = '정의형', 'grade_rule C*: 포함 결정형이나 원점 시행 코드·단위만 쓰고 비선택 근거 있음 — ' + r.근거
    elif r.등급A == 'NA' and r.등급B == 'E': t, why = '정의형', 'grade_rule NA: 공식 모형·기준선 밖 — ' + r.근거
    elif r.요소ID == 'T3-02': t, why = '정의형', '레이블 값의 분류 빈티지(값)는 C, 셀 존재·적격은 T3-01·03(B2)으로 분리(범위 재기술) — ' + r.근거
    elif r.요소ID == 'T1-07': t, why = '사실형', '이름 경로는 레이블 값 오귀속(양수 법인)이고 관측 가능 여부는 코드 경로(원출원인 코드)로도 보장 → C; 정량 Δ +0.002~+0.005(C16-T1) 문턱 미만 — ' + r.근거
    else: t, why = '사실형', r.근거
    adj.append(dict(요소ID=r.요소ID, 코더A=r.등급A, 코더B=r.등급B, 유형=t, 최종=r.등급A, 조정근거=why))
adj = pd.DataFrame(adj)

# ---------------- 원고(v18) 공개 재판정: B·C 행 ----------------
DISC = {
 'T1-01': ('공개됨', '229(조건·진입 모두 2025-08까지), 410(T1 위험쌍 조건 2025-08, 0.461–0.468), 79(T1 위험쌍 정의)'),
 'T1-02': ('부분', '229·410(조건의 사후성 일반); 신규진입 법인 집합(applicant_query) 기제 미기재 — 미정량(C17-T1에서 측정 중)'),
 'T1-03': ('공개됨', '229, 410, 79'),
 'T1-04': ('공개됨', "229('공개 선행 출원' + 조건 2025-08), 410"),
 'T1-05': ('부분', "79('2026 권리목록' 값 예외만); 조회 실패 법인 제외 미기재"),
 'T1-07': ('부분', "79('명칭' 입력값 예외), 229('코드·이름 매칭 식별'); 레이블 경로의 현재 명칭 성격 미기재"),
 'T1-08': ('공개됨', '79, 143'), 'T1-09': ('공개됨', '79, 143'), 'T1-10': ('공개됨', '79, 143'), 'T1-11': ('공개됨', '79, 143'),
 'T1-12': ('부분', "410('DART 업종은 2026년 스냅샷' 일반); T1 기준선 사용 미기재. 143 '법인 연결표를 빼면 시점을 맞추거나'의 예외에도 빠짐"),
 'T1-15': ('부분', "79('라벨' 현행 값 = 입력값 예외); 위험집합·레이블 결정 역할 미기재"),
 'T3-01': ('공개됨', "217(표 6 '신설 IPC 셀 제외 0.406', 재분류 소급 셀), 409(H10 소급, 빼거나 합쳐도 0.39–0.42 → V1r 0.380으로 범위 대체 필요); 79 예외 목록에는 없음"),
 'T3-02': ('공개됨', "409('부상 순증가의 45%가 H10 셀'), 217"),
 'T3-03': ('부분', '409(소급 일반); 기존 코드 셀의 적격 이동(6셀) 미기재'),
 'T3-04': ('부분', '409(KR 과거 문서 소급); 피처 창 역할 미기재'),
 'T3-05': ('부분', "79('라벨'); 셀 KSIC 축 결정 역할 미기재"),
 'T3-06': ('공개됨', "79('광공업지수'), 143"), 'T3-07': ('공개됨', "79('RoS')"), 'T3-09': ('공개됨', "79('명칭')"),
 'T4-02': ('부분', "79('라벨')"), 'T4-05': ('공개됨', "79('서지')"), 'T4-06': ('공개됨', "79('광공업지수')"),
 'T4-07': ('공개됨', "79('서지'; family 필드를 서지로 해석)"), 'T4-08': ('공개됨', "79('명칭')"),
 'N4-01': ('공개됨', "143('법인 연결표를 빼면'), 410"), 'N4-02': ('공개됨', "79('명칭'), 410"), 'N4-05': ('공개됨', '410'),
 'N4-06': ('공개됨', '127(표 4), 143'), 'N4-07': ('공개됨', "79('NTIS')"),
 'N4-08': ('부분', "410('DART 업종 2026 스냅샷'); 재무 대상 목록 미기재"), 'N4-10': ('부분', "79('라벨')"),
 'N5-04': ('공개됨', "79('서지')"), 'N5-05': ('공개됨', "79('명칭')"), 'N5-06': ('공개됨', "79('NTIS')"), 'N5-07': ('공개됨', "79('RoS')"),
 'N5-08': ('공개됨', "79('서지')"), 'N5-10': ('부분', "79('라벨')"),
 'N6-01': ('부분', "79('NTIS' 값 예외), 257(과제 수 공식 일치); 모집단 스냅샷 역할 미기재"),
 'N6-02': ('부분', "143('법인 연결표'), 410(연결 편향 일반); N6 미연결=결과 0 미기재(E-C16T2-04 미반영)"),
 'N6-04': ('부분', "79('명칭' 입력값 예외), 267(동명 복원 Δ); 레이블 이름 경로의 현재 권리자 명칭 성격 미기재 [verify_0 수정]"),
 'N6-05': ('공개됨', "79('NTIS')"), 'N6-06': ('공개됨', "143('국민연금 사업장은 상호 변경 키를 별칭 보정')"), 'N6-08': ('부분', "79('라벨')"),
 'N7-01': ('공개됨', '143, 275(70%만 0.18), 410'), 'N7-02': ('부분', "79('NTIS' 값 예외), 275(보안과제); 제외 규칙 역할 미기재"),
 'N7-04': ('공개됨', '127, 143, 275'), 'N7-05': ('공개됨', "79('NTIS')"), 'N7-07': ('공개됨', "79('명칭'), 410"), 'N7-08': ('부분', "79('라벨')"),
 'N8-02': ('부분', "79('라벨')"), 'N8-04': ('공개됨', "79('서지')"), 'N8-05': ('공개됨', "79('명칭')"), 'N8-07': ('공개됨', "79('서지')"),
 'N9-01': ('부분', "79('NTIS'), 257; 단위 스냅샷 역할 미기재"), 'N9-03': ('공개됨', "79('NTIS'), 265"), 'N9-04': ('공개됨', "79('NTIS')"),
 'N9-05': ('공개됨', "79('명칭')"), 'N9-07': ('부분', "79('라벨')"),
 'X-01': ('미공개', "-(73행 표 2 주 '상속' 언급만)"),
}
DELTA = {
 'T1-01': 'C16-T1 원점 가시 모집단 S1-full −0.0119, S1-fixed −0.0056, S2 −0.0014(T1-01·03·04 공동)',
 'T1-03': 'T1-01과 공동(C16-T1)', 'T1-04': 'T1-01과 공동(C16-T1)',
 'T1-02': '미정량(C17-T1에서 측정 중)',
 'T3-01': f"최종 V1 {sv('V1','final'):+.4f}, V1r {sv('V1r','final'):+.4f}, V2 {sv('V2','final'):+.4f}; 개발 평균 V1 {sv('V1','dev_mean'):+.4f}, V1r {sv('V1r','dev_mean'):+.4f}, V2 {sv('V2','dev_mean'):+.4f}(C17-T3 t3_newipc_sens.csv)",
 'T3-02': 'V2에서 기존 셀 레이블 25개 변화(V2 Δ에 포함, 분리 불가)',
 'T3-03': 'V2 Δ에 포함(6셀 적격 이동, 분리 불가)',
}
new = old.copy()
new = new.merge(ca[['요소ID', '등급A', '하위', '근거']], on='요소ID').merge(cb[['요소ID', '등급B']], on='요소ID')
new['등급_C16'] = new['등급'].fillna('NA'); new['등급'] = new['등급A']
new['등급하위'] = new['하위']
fix_el = {'T3-01': '적격 셀 집합(단위 존재·적격): 5개국 공개연도 t0−2..t0 문서 합 ≥50(r3core.py:19 MIN_TOTAL=50), 셀 정의는 현행 분류 — 90셀(1.8%)이 IPC 2021.01에 없는 메인그룹',
          'T3-02': '레이블 값: KR 공개 2022–24 대 2019–21 로그 증가 상위 10%(두 창 모두 현행 분류로 계수)',
          'T3-03': '재분류에 따른 기존 코드 셀의 적격 변화(문서가 신설 코드로 이동해 원점 시행판 코드 셀 6개가 적격 집합에서 빠짐)',
          'T3-04': '피처 창(원점 이전 문서)의 현행 분류 배정'}
for k, v in fix_el.items(): new.loc[new.요소ID == k, '요소'] = v + ' [범위 재기술 C17-T3]'
new.loc[new.요소ID == 'T3-01', '근거(파일:행)'] = r"(MC)\T3_기술셀부상예측\R3\r3core.py:19,26(MIN_TOTAL=50 적격·레이블); r3lib.py:59-61(eligible), 98·142·149(≥10은 사전 필터); C17-T3 t3_newipc_sens.py U 집합"
new['원고_공개판정_v18'] = new.요소ID.map(lambda k: DISC[k][0] if k in DISC else None)
new['원고_줄_v18'] = new.요소ID.map(lambda k: DISC[k][1] if k in DISC else None)
# B·C가 아닌 행은 C16 판정을 이어받되 v18 줄 변화(410 NPS 문장 삭제) 반영
for i, r in new.iterrows():
    if r.등급 in ('B', 'C'): assert r.요소ID in DISC, r.요소ID
    elif r.요소ID in DISC: pass
    else:
        new.at[i, '원고_공개판정_v18'] = r['원고_공개판정']; new.at[i, '원고_줄_v18'] = r['원고_줄']
for k, v in DELTA.items(): new.loc[new.요소ID == k, 'Δ값'] = v
new['범위'] = np.where(new.요소ID.str.startswith('X'), '트랙 밖', '트랙')
cols = ['범위', '트랙', '요소ID', '요소종류', '요소', '등급', '등급하위', '등급_C16', '등급A', '등급B', '근거(파일:행)', '근거', '트랙문서_공개', '원고_공개판정_v18', '원고_줄_v18',
        '원고_공개판정', '원고_줄', '기존정량화(F행·C번호)', 'Δ값', '미정량', '비고']
new = new[cols].rename(columns={'원고_공개판정': '원고_공개판정_C16(v17)', '원고_줄': '원고_줄_C16(v17)', '근거': '코더A_속성근거'})
new.to_csv(os.path.join(W, 'pit_map_v2.csv'), index=False, encoding='utf-8-sig')
ca.to_csv(os.path.join(W, 'coder_a.csv'), index=False, encoding='utf-8-sig')
cb.to_csv(os.path.join(W, 'coder_b.csv'), index=False, encoding='utf-8-sig')
adj.to_csv(os.path.join(W, 'adjudication.csv'), index=False, encoding='utf-8-sig')

# ---------------- 집계 ----------------
rows = []
for scope, d in [('트랙86', new[new.범위 == '트랙']), ('트랙밖2', new[new.범위 == '트랙 밖']), ('전체88', new)]:
    for trk, g in list(d.groupby('트랙')) + [('합계', d)]:
        r = dict(범위=scope, 트랙=trk, n=len(g))
        for gr in ['A', 'B', 'C', 'D', 'E', 'NA']: r[gr] = int((g.등급 == gr).sum())
        r['B1'] = int((g.등급하위 == 'B1').sum()); r['B2'] = int((g.등급하위 == 'B2').sum()); r['B3'] = int((g.등급하위 == 'B3').sum()); r['C*'] = int((g.등급하위 == 'C*').sum())
        bc = g[g.등급.isin(['B', 'C'])]
        for gr in ['B', 'C']:
            x = bc[bc.등급 == gr]
            r[f'{gr}_공개됨'] = int((x.원고_공개판정_v18 == '공개됨').sum()); r[f'{gr}_부분'] = int(x.원고_공개판정_v18.str.startswith('부분').sum()); r[f'{gr}_미공개'] = int((x.원고_공개판정_v18 == '미공개').sum())
        rows.append(r)
gt = pd.DataFrame(rows)
gt = gt[~((gt.범위 == '전체88') & (gt.트랙 != '합계'))]
gt.to_csv(os.path.join(W, 'grade_table.csv'), index=False, encoding='utf-8-sig')
chg = new[new.등급 != new.등급_C16][['요소ID', '등급_C16', '등급', '등급하위']]
out = dict(kappa88=k88, po88=po88, kappa86=k86, po86=po86, n_disagree=int(len(adj)), disagree_types=adj.유형.value_counts().to_dict(),
           disagree_pairs=adj.groupby(['코더A', '코더B']).size().to_dict().__repr__(),
           changed_vs_C16=chg.to_dict('records'), kappa_vs_C16=kappa(new.등급.tolist(), new.등급_C16.tolist()))
json.dump(out, open(os.path.join(W, 'cache', 'map_v2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print(gt.to_string())

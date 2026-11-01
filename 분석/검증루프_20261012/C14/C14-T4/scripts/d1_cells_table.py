# C14-T4 d1: 칸 단위 판정 표 cells_verdict.csv (수치는 cache/*.json, se_struct.json, verify_0에서 읽음)
import sys, os, json, csv
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib14 import *
se = json.load(open(W + r'\se_struct.json', encoding='utf-8'))
p0c = json.load(open(C + r'\c2_p0c.json', encoding='utf-8'))
b1 = json.load(open(C + r'\b1_out.json', encoding='utf-8'))
b3 = json.load(open(C + r'\b3_p0b.json', encoding='utf-8'))
v1 = json.load(open(W + r'\verify_0\v1_out.json', encoding='utf-8'))
v2 = json.load(open(W + r'\verify_0\v2_summary.json', encoding='utf-8'))
H = ['cell', 'unit', 'event', 'definition', 'C1', 'C2', 'C3', 'C4', 'C5', 'P0a', 'P0b', 'P0c', 'verdict', 'first_fail', 'evidence']
R = []
pw = se['P0b']['min_power_primary']; sep = se['dAUC_single']['SE_primary']; spr = se['dAUC_single']['SE_prod']
R.append(['A:C12-T4', '(나라, 출원인, 서브클래스)', 'Ea(라벨 무관 기술 진입) x KSIC 맥락 피처',
          'CN·EP 활성 출원인의 자기에게 새 IPC 서브클래스 첫 공개(다음 반기); 제공 KSIC 맥락 대 공개 배정 맥락',
          '약한 통과(결과 라벨 무관, KSIC는 피처; 모집단은 제공 50 KSIC 범위)', '해당 없음(결과가 라벨 정의 아님) - P0(c) 포함',
          '통과(CN·EP 2025-H1 단위 결과 미생성; 개발 D 노출)', '[경고] 공개 외부 데이터 결합 없음(5개국 제공 추세만)',
          '[경고] C13 묶음 규칙상 원고 근거 B(MS-B)와 같은 F10(나라만 다름)',
          '통과(잠금 예상 15,961; C12)',
          f"통과(구조=교차; 주 SE(CGM) 단일 원점 {min(sep):.5f}-{max(sep):.5f}, 최소 검정력 {pw:.3f}; 곱 가중 원판 SE {min(spr):.5f}-{max(spr):.5f})",
          f"기각(c-ii R_ctx {p0c['c_ii']['R_ctx']:.3f} >= 0.50; c-i 통과 E_prov->pub {p0c['c_i']['E_prov_to_pub']:.4f} 대 E_pub->prov {p0c['c_i']['E_pub_to_prov']:.4f})",
          '기각(P0(c))', 'P0(c-ii)',
          f"se_struct.json; cache/c2_p0c.json; 반박(verify_0): XGB 귀무 R_ctx {v2['R_ctx_xgb_mean5']:.3f}(<0.50, 판정 뒤집힘), 로지스틱 재학습 P0(b) 검정력 {v1['A_logit_2024']['power_lock_target']:.3f}(<0.80)"])
eb = b1['Eb_pos']; en = b1['Eb_neg']; ebs = b1['Eb_pos_stable']; ens = b1['Eb_neg_stable']
R.append(['U6xEb', '(나라, 조직 쌍)', 'Eb 중단', '활성 이종 산업 관계(3년 공동 >=2, 1년 >=1, 제공 최빈 KSIC 상이)의 다음 반기 공동 문서 0건(CN·EP)',
          '약한 통과(이종 산업 위험집합을 제공 KSIC로 정의; 공동출원인 연결; 5개국 추세 피처)', '설계 반영(위험집합이 라벨 정의 -> P0(c) 역방향·귀무)',
          '통과(CN·EP 2025-H1 관계 결과 미집계)', "약한 통과(서류: X1' HS6 수입 다리 재사용 가능; P0(b) 모형에는 미포함)", '통과(규칙상 U6xEa와 다른 칸)',
          f"조건부 통과(전체: 개발 중단 {eb['dev_sum']:,}/지속 {en['dev_sum']:,}, 잠금 예상 {eb['lock_exp']:,.0f}/{en['lock_exp']:,.0f}; 안정 패널 잠금 {ebs['lock_exp']:,.0f}/{ens['lock_exp']:,.0f} < 1,000)",
          f"실패(dAUC {b3['dAUC_prov_pub']:+.4f}, 주 SE(KSIC dyad) {b3['SE_dev']['SE_primary']:.4f}, 검정력 {b3['power']['SE_primary']:.3f}; iid SE로도 {b3['power']['SE_iid']:.3f})",
          '미실행', '기각', 'P0(b)',
          f"cache/b1_out.json, b3_p0b.json; verify_0 v1: 영향함수 dyad 검정력 {v1['B_U6Eb_stored_if']['power_lock_primary']:.3f}, 로지스틱 {v1['B_U6Eb_logit']['power_lock_primary']:.3f}"])
ec = b1['Ec_pos']
R.append(['U6xEc', '(나라, 조직 쌍)', 'Ec 증감(강화)', '활성 이종 산업 관계의 다음 반기 공동 문서 >=2 그리고 직전 3년 반기 평균 초과(CN·EP)',
          '약한 통과(U6xEb와 같음)', '설계 반영', '통과', '약한 통과(서류)', '통과(규칙상 다른 칸)',
          f"실패(개발 {ec['dev_sum']:,} 통과, 잠금 예상 {ec['lock_exp']:,.0f} < 1,000; 안정 패널 {b1['Ec_pos_stable']['lock_exp']:,.0f})",
          '미실행', '미실행', '기각', 'P0(a)', 'cache/b1_out.json, b1_by_origin.csv'])
R.append(['U6xEe', '(조직 쌍)', 'Ee 외부 사건', '공동출원 상대의 폐업·피합병·공동 정부R&D 수주 등',
          '실패(사건이 기업 단위 외부 사건; KSIC는 위험집합 범위뿐이라 지워도 같은 사건)', '-',
          '보조: KR 외부 사건 2025-(NTIS 키 필요), CN·EP 조직 외부 사건의 무로그인 연결 원천 없음', '-',
          '보조 실패(F6 기업 외부 사건의 쌍 표현)', '-', '-', '-', '기각', 'C1', 'inventory.csv F6(17행); C13 candidates A5'])
eg = b1['Eg_CN']
R.append(['U6xEg', '(나라, 조직)', 'Eg 국경 간 관계 형성',
          "C1 충족을 위해 이종 산업 조건 포함 = X1' 첫 관계 중 상대가 외국 조직(CN: 상대 키에 한자 없음; KR: 등록원부 코드 5; EP: 국적 정보 없음)",
          '약한 통과(KSIC 조건 포함 시)', '설계 반영', '통과', '약한 통과', "경계(규칙상 다른 칸이나 사건은 F11 X1'의 부분집합)",
          f"실패(CN 개발 {eg['dev_sum_firms']}기업, 잠금 예상 {eg['lock_exp']:.1f}; 외국 상대 비율 0.1-0.5%; KR은 창 안 조직 공동 등록 전체 797-1,183건이 상한; EP 정의 불가)",
          '미실행', '미실행', '기각', 'P0(a)', 'cache/b1_out.json(Eg_CN); C13 cache/s2_p0a_by_origin.csv'])
R.append(['U6xEd', '(조직 쌍)', 'Ed 거래', '조직 간 권리 이전·실시 관계(산업 간)', '약한 통과(양쪽 제공 라벨)', '설계 필요', '통과(등록원부 2025-)', '-',
          "실패(F2 C07-X 산업 간 권리이전·C08-A' 산업 간 실시권과 같은 묶음)", "보조 실패(C07-X 잠금 예상 478, C08-A' 490)", '-', '-', '기각', 'C5', 'inventory.csv F2'])
for cell, ev, dfn, c5 in [
        ('U4xEb', 'Eb 중단', '(IPC x KSIC) 활성 셀의 다음 창 0건(셀 소멸)', '보조 실패(F5 T3·C01-C-B의 방향 반대 변형)'),
        ('U4xEd', 'Ed 거래', '셀 단위 권리 이전·실시 총량', '보조 실패(F2의 셀 집계)'),
        ('U4xEe', 'Ee 외부 사건', '셀 단위 외부 결과(외부 통계는 KSIC·HS 단위라 IPC 차원 없음 -> U5xEe로 환원)', '보조 실패(F8)'),
        ('U4xEg', 'Eg 국가 간 확산', '해외 활성 셀의 다른 나라 첫 등장', "보조 실패(F5 C01-C-B '해외 부상 셀의 한국 첫 공개'와 같은 칸; P1 기각 합 371)")]:
    R.append([cell, '(IPC, KSIC) 셀', ev, dfn, '실패(결과 정의에 기업(출원인) 연결 불필요)', '-', '-', '-', c5, '-', '-', '-', '기각', 'C1', 'C13 PREREG §2 C1; inventory.csv'])
for cell, ev, dfn, c5 in [
        ('U5xEb', 'Eb 쇠퇴·소멸', 'KSIC(x나라) 출원·생산의 감소·소멸', '보조 실패(F5 T2·F8 T6의 방향 변형)'),
        ('U5xEd', 'Ed 거래', 'KSIC 단위 거래 총량', '보조 실패(F2 집계)'),
        ('U5xEe', 'Ee 외부 결과', 'KSIC 고용·출하·수출 등', '보조 실패(F8 N1-N3·T6·R8 P2)'),
        ('U5xEg', 'Eg 나라 간 이동', 'KSIC 몫의 나라 간 이동', '보조 실패(F5 R2-G3·T2)')]:
    R.append([cell, 'KSIC(x나라)', ev, dfn, '실패(기업 연결 불필요)', '-', '-', '-', c5, '보조 실패(구조 상한: 단위 <= 50 x 5 = 250/잠금 창)', '-', '-', '기각', 'C1', 'C13 PREREG §2; R2 N1·N2 최종 실패'])
for cell, unit, ev, dfn, c5 in [
        ('U1xEa', '문서·특허', 'Ea 첫 발생', '특허의 첫 이전·첫 실시·첫 분쟁 등', '보조 실패(F2 T5·C08-A, F7 N8)'),
        ('U1xEb', '문서·특허', 'Eb 중단', '등록특허 불납 포기·소멸', '보조 실패(F7 T4·R2-firm-LAPSE)'),
        ('U1xEc', '문서·특허', 'Ec 증감', '권리 일부포기·청구항 변화·선납', '보조 실패(F7 C08-B·C08-D)'),
        ('U2xEa', '기업', 'Ea 첫 발생', '기업의 첫 특허·첫 제공범위 출원', '보조 실패(F6 N6·R3-I1·C04-B->C05)'),
        ('U2xEb', '기업', 'Eb 중단', '폐업·NPS 탈퇴·피합병', '보조 실패(F6 N4·R3-I2·R5-J3)'),
        ('U2xEc', '기업', 'Ec 증감', '고용·매출·보험료 급증', '보조 실패(F6 R3-I3·C04-A·C04-C·R2-firm-HG)'),
        ('U2xEd', '기업', 'Ed 거래', '기술 구매(양수) 기업', '보조 실패(F2 R2-firm-BUY)')]:
    R.append([cell, unit, ev, dfn, '실패(결과가 제공 KSIC 없이 정의)', '-', '-', '-', c5, '-', '-', '-', '기각', 'C1', 'inventory.csv; C13 candidates A5·A6'])
with open(W + r'\cells_verdict.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.writer(fh); w.writerow(H); w.writerows(R)
out('rows', len(R))

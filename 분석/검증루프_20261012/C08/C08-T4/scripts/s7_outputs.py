# C08-T4 s7: 결과 CSV 작성(candidates, feasibility, bias_checks, essentiality_null). 수치는 cache·verify_0 산출에서 읽는다.
import sys, os, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import C, W
J = lambda f: json.load(open(os.path.join(C, f), encoding='utf-8'))
p0 = J('p0b_out.json'); s1 = J('s1_log.json'); s1c = J('s1c_monthly.json'); s5 = J('s5_p5_log.json')
v1 = json.load(open(os.path.join(W, 'verify_0', 'v1_out.json'), encoding='utf-8'))
v2 = json.load(open(os.path.join(W, 'verify_0', 'v2_out.json'), encoding='utf-8'))

# 1) candidates.csv
cand = [
    ['A', '실시권 신규 설정(문서)', '창 안 전용·통상실시권 시작일이 있는 생존 특허', 'P', 'T5 기술거래(실시권은 T5 피처·사건 범주와 부분 겹침), C03-B(실시권 관련 후보) 확인', '부분 겹침', 'P0 기각(기저율 0.134%, 잠금 543)'],
    ["A'", '산업 간 실시권(문서)', 'A ∧ 비개인 실시권자 ∧ 권리자·출원인과 다름 ∧ 실시권자 포트폴리오에 특허 KSIC 없음', 'S', 'T5·C07-X(산업 간 권리이전)의 실시권판', '정의상 새로움(상한부터 미달)', 'P0 기각(상한 0.117%, 잠금 490)'],
    ['B', '권리 일부포기(문서)', "창 안 '권리의 일부말소등록'(사유에 '포기')", 'P', 'T4(전부 불납 포기)와 다른 사건(청구항 일부 포기). _ideas_R5–R8·C01–C07에 없음', '새로움', 'P0 기각(기저율 0.311% 통과, 잠금 774 < 1,000; 2016 스파이크 1.06%)'],
    ['C', '불납 소멸 특허의 회복', 'Y년 불납 소멸 특허의 365일 내 회복등록', 'P', 'C03이 구조 이유로 파일럿 전 제외한 후보', '중복', 'P0 기각(기저율 0.002%; 등록원부가 회복 시 소멸·회복을 같은 날 기록하는 형식 탓)'],
    ['D', '연차료 다년 선납(문서)', '창 안 statAnnl ≥ 4 이고 2개 연차 이상 한 번에 납부', 'P', 'T4·T5·N4의 피처(다년 일괄납부)로만 쓰임, 결과로는 없음', '새로움(피처 겹침)', 'P0 기각(기저율 0.276% < 0.30%, 잠금 1,145)'],
    ['E', '연차료 지연 납부(엄격, 납부일 > 기한)', '창 안 statAnnl ≥ 4 납부행 중 납부일 > 기한', 'P', 'T4(불납 포기)와 같은 연차료 결정의 다른 결말; T4·N4 피처(지연납부 횟수)', '부분 겹침(결과로는 새로움)', 'P0 통과(4.449%, 잠금 15,888) → 주말·공휴일 합법 납부 혼입으로 E*로 좁힘'],
    ['E*', '연차료 지연(추가료 구간) 납부 — 주 후보', '창 안 statAnnl ≥ 4 납부행 중 납부일 − 기한 > 7일', 'P(+F: M1 산업 피처)', '위와 같음. windows_used·_ideas_R5–R8·C01–C07 summary에서 지연 납부를 결과로 쓴 과제 없음', '부분 겹침(결과로는 새로움)', 'P1–P3 통과, P4 기각'],
    ['X1', '질권(IP 담보) 설정', '—', '—', '등록원부에 질권 사건 유형 없음(owner 유질계약 이전 15행)', '—', '0단계 제외(데이터 없음)'],
    ['X2', '정정(심판)예고·존속기간연장', '—', '—', '연 18–38건 / 0–4건', '—', '0단계 제외(기준 미달 명백)'],
    ['X3', '기술평가 예고·확정', '—', '—', '실용신안 전용, 2010년 이후 연 ≤44건', '—', '0단계 제외'],
    ['X4', 'KR 출원의 첫 해외 패밀리 진입', '—', '—', 'R6 K1·K1P, R3 I4, C03-A와 겹침', '중복', '0단계 제외'],
    ['X5', '동종 산업 기업으로의 권리이전 / 비주력 산업 특허 포기·매각', '—', '—', 'C07-X 여집합·T5 사건 부분집합 / T4·T5 사건과 같음', '중복', '0단계 제외'],
]
pd.DataFrame(cand, columns=['id', 'name', 'event', 'ksic_structure(S/P/F)', 'overlap_checked_against', 'overlap', 'status']).to_csv(os.path.join(W, 'candidates.csv'), index=False, encoding='utf-8-sig')

# 2) feasibility.csv
fr = []
for f in p0['feas']:
    fr.append({'stage': 'P0', 'cand': f['cand'], 'dev_pos_total': f['dev_pos_total'], 'dev_rate_mean': round(f['dev_rate_mean'], 6), 'dev_rate_min': round(f['dev_rate_min'], 6),
               'dev_rate_max': round(f['dev_rate_max'], 6), 'rate_2021_22': round(f['rate_2021_22'], 6), 'lock_risk': round(f['lock_risk'], 1),
               'lock_exp_main': round(f['lock_exp_main'], 1), 'lock_exp_alldev': round(f['lock_exp_alldev'], 1), 'pass': f['P0'],
               'verify_0': ''})
for r in fr:
    k = {'E': 'E0', 'A': 'A', 'B': 'B', 'D': 'D'}.get(r['cand'])
    if k:
        vv = v1['summary'][k]
        r['verify_0'] = f"v1 독립 SQL: 합 {vv['pos_total']}, 잠금 예상 {vv['lock_exp_main']:.1f} ({'일치' if vv['pos_total'] == r['dev_pos_total'] else '불일치'})"
P1 = s1['P1']
fr.append({'stage': 'P1', 'cand': 'E*(지연 > 7일)', 'dev_pos_total': P1['pos_total'], 'dev_rate_mean': round(P1['dev_rate_mean'], 6),
           'dev_rate_min': round(min(P1['rate_by_origin'].values()), 6), 'dev_rate_max': round(max(P1['rate_by_origin'].values()), 6),
           'rate_2021_22': round(P1['rate_2021_22'], 6), 'lock_risk': P1['lock_risk'], 'lock_exp_main': round(P1['lock_exp_main'], 1),
           'lock_exp_alldev': round(P1['lock_exp_alldev'], 1), 'pass': P1['pass_a'] and P1['pass_b'],
           'verify_0': f"v1: 합 {v1['summary']['E7']['pos_total']}, 원점별 최소 {v1['summary']['E7']['min_pos']}, 잠금 {v1['summary']['E7']['lock_exp_main']:.1f}; 고유 특허 {P1['unique_patents']}; 원점별 {json.dumps(P1['pos_by_origin'])}"})
fr.append({'stage': 'P1(보조)', 'cand': 'E3(지연 > 3일)', 'dev_pos_total': v1['summary']['E3']['pos_total'], 'dev_rate_mean': round(P1['E3']['rate_mean'], 6), 'dev_rate_min': '', 'dev_rate_max': '',
           'rate_2021_22': '', 'lock_risk': P1['lock_risk'], 'lock_exp_main': round(P1['E3']['lock_exp_main'], 1), 'lock_exp_alldev': '', 'pass': '보고만',
           'verify_0': 's1와 원점 4곳에서 ±1(2월 29일 기한 처리 차이)'})
s4 = pd.read_csv(os.path.join(C, 's4_p4_result.csv'))
for _, r in s4.iterrows():
    fr.append({'stage': 'P4', 'cand': f"E* {r.model} seed {r.seed}", 'dev_pos_total': '', 'dev_rate_mean': '', 'dev_rate_min': '', 'dev_rate_max': '', 'rate_2021_22': '',
               'lock_risk': '', 'lock_exp_main': '', 'lock_exp_alldev': '',
               'pass': bool(r.pass_seed) if r.model == 'XGB' else '판정 밖',
               'verify_0': f"AUC M0 {r.auc_M0_mean:.4f} M1 {r.auc_M1_mean:.4f}; Δskill {r.dskill_mean:+.4f} [{r.dskill_ci_lo:+.4f}, {r.dskill_ci_hi:+.4f}]; PR-AUC {r.prauc_M0_mean:.4f}→{r.prauc_M1_mean:.4f}"})
for s in (5, 6):
    o = json.load(open(os.path.join(W, 'verify_0', f'v4_out_{s}.json')))
    fr.append({'stage': 'P4 반박(verify_0 v4)', 'cand': f'E* HGB seed {s}', 'pass': '판정 밖', 'verify_0': f"Δskill {o['dskill_mean']:+.4f} [{o['ci_lo']:+.4f}, {o['ci_hi']:+.4f}]"})
pd.DataFrame(fr).to_csv(os.path.join(W, 'feasibility.csv'), index=False, encoding='utf-8-sig')

# 3) bias_checks.csv
b = []
add = lambda chk, val, crit, res, note='': b.append({'check': chk, 'value': val, 'criterion': crit, 'result': res, 'note': note})
a = s1['P2a']; add('P2(a) 모집단 대조', f"kr_registered {a['kr_n']} ↔ registry {a['reg_n']}; 양방향 일치 {a['kr_in_reg']:.4f}/{a['reg_in_kr']:.4f}; KSIC 일치 {a['ksic_agree']:.4f}", '≥ 0.99', '통과', '등록원부는 제공 50개 KSIC 문서에 대해서만 수집 — 전체 KR 등록특허가 아님(범위 한계)')
bb = s1['P2b']; add('P2(b) 납부 기록 완결성', '; '.join(f"{k}:{v['covered']:.4f}(n={v['n']})" for k, v in bb['by_origin'].items()), '원점 모두 ≥ 0.95, KSIC 최솟값 < 0.90 경고', '통과', f"KSIC 최솟값 {bb['ksic_min']:.4f} {bb['ksic_min_where']}")
c = s1['P2c']; add('P2(c) 중복·결측', f"pay (rgst_no,statAnnl) 중복 {c['dup']}, 납부일 형식 오류 {c['bad_date']}, 등록일 결측 {c['rgst_null']}", '각 ≤ 1%', '통과')
d = s1['P2d']; add('P2(d) 집중도', f"상위 10 출원인 몫 {d['top10_cd_share']:.4f}; KSIC 출현 {d['ksic_present']}; 상위 5 KSIC 몫 양성 {d['top5_ksic_share_pos']:.3f} vs 위험 {d['top5_ksic_share_risk']:.3f}", '> 0.50 실패, > 0.30 경고', '통과',
    f"KSIC별 기저율 {d['ksic_rate_min']:.4f}–{d['ksic_rate_max']:.4f}")
add('P2(d) 출원인 유형 분포', '; '.join(f"{k}: 기저율 {d['type_rate'][k]:.4f}, 양성몫 {d['type_share_pos'].get(k, 0):.3f}, 위험몫 {d['type_share_risk'].get(k, 0):.3f}" for k in sorted(d['type_rate'])), '보고', '기록',
    '1 국내법인 2 대학 3 연구기관 4 국내개인 5 외국법인 6 외국개인. 개인(8.8%)·대학(4.6%)이 높고 외국법인(0.35%)은 낮다 — 대리인 납부 관행 차이. 국내 개인 위험몫 8.2%가 양성 29.7%')
add('P2(e) 기저율 안정성', f"최고/최저 {P1['rate_max_min_ratio']:.3f} (2015 1.885% → 2021 2.638%)", '> 2.0 경고', '통과', '2015→2017 상승; 원인 미확인(추가납부 가산금 제도·전자납부 변화 가능성)')
p3b = s1['P3b']; add('P3(b) 소급 기록', f"불납 소멸 n={p3b['n']}, 커버 종료→등록 ≤183일 {p3b['share_gap_le_183']:.4f}, 중위 {p3b['median_gap']}일", '< 5%', '통과', '소멸등록 날짜는 처리일(추가납부 6개월 + 회복 3개월 뒤)')
add('P3(c) 잠금 관측 가능성', f"pay 총 행 2023 {s1c['year_tot']['2023']}, 2024 {s1c['year_tot']['2024']}, 2025 {s1c['year_tot']['2025']}, 2026(1–8월) {s1c['year_tot']['2026']}; 2025/2024 {s1c['ratio_2025_2024']:.3f}; 2026-01~07 월평균/2025 {s1c['mon26_1to7_over_mon25']:.3f}",
    '≥ 0.85 그리고 ≥ 0.5', '통과', '2026-08은 4,070행(스냅샷 중순 절단). 단위 수준 결과 미조회')
od = s1['P3d']; add('P3(d) as-of 연체 상태', '; '.join(f"{k}: 양성 중 연체 {v['pos_overdue_share']:.3f}, 위험 중 {v['risk_overdue_share']:.3f}, 기저율 연체 {v['rate_if_overdue']:.3f}/비연체 {v['rate_if_not']:.3f}" for k, v in od.items()), '보고', '기록',
    '연체 중이어도 지연 납부율은 6–10%뿐(나머지는 불납 소멸) — T4 결과와 같은 결정의 다른 결말')
add('P0-c 주말 기한 효과', '엄격 정의 지연률 토 9.95%·일 11.13% vs 평일 5.5–6.4%; 지연 > 3일이면 4.10–4.33%로 요일 무관', '정의 점검', 'E* 채택', '특허법 제14조 기간 말일 규칙')
add('P0-c 금액 흔적', '지연 납부 금액 = 같은 (연도, 연차) 중위의 0.58–0.67배(정시 1.02배)', '정의 점검', '추가료 확인 불가', '감면 대상(개인·중소기업) 비중 차이로 보임 — 추가료 포함 여부를 금액으로 검증할 수 없음')
pd.DataFrame(b).to_csv(os.path.join(W, 'bias_checks.csv'), index=False, encoding='utf-8-sig')

# 4) essentiality_null.csv
e = pd.read_csv(os.path.join(C, 's5_p5_result.csv'))
e['p5b_auc_prov_ci'] = f"[{s5['P5b']['ci_lo']:.4f}, {s5['P5b']['ci_hi']:.4f}]"
for unit in ['cd', 'ksic']:
    e[f'refute_{unit}_boot_delta_ci'] = e['assign'].map(lambda A: (lambda r: f"[{r['delta_ci_lo']:.4f}, {r['delta_ci_hi']:.4f}] (≤0 비율 {r['share_le0']:.2f})" if r else '')(v2.get(f'ii_{unit}_{A}')))
pl = json.load(open(os.path.join(C, 'p4_log_pub.json')))
pv = json.load(open(os.path.join(C, 'p4_log_20261012.json')))
e['p5c_model_auc_pub_mean'] = e['assign'].map(lambda A: round(np.mean([pl[f'{v}_M1_{A}_auc'] for v in range(2018, 2023)]), 5) if f'2018_M1_{A}_auc' in pl else '')
e['p5c_model_auc_prov_mean'] = round(np.mean([pv[f'{v}_M1_auc'] for v in range(2018, 2023)]), 5)
e['verdict'] = '판정 밖(P4 기각 뒤 보고)'
e.to_csv(os.path.join(W, 'essentiality_null.csv'), index=False, encoding='utf-8-sig')
print('ok')

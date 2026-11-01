# audit_numbers.csv 생성. 원천 값은 이번 바퀴에 원천 파일을 직접 열어 읽은 값(키 경로 기록), 재계산 값은 recompute_results.json.
import csv, json, sys
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
W = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C01/C01-T2/"
RC = json.load(open(W + "recompute_results.json", encoding='utf-8'))


def dec_places(s):
    s = s.replace('%', '').replace('p', '')
    return len(s.split('.')[1]) if '.' in s else 0


def judge(ms, src, scale=1.0):
    d = dec_places(ms)
    v = Decimal(repr(float(src))) * Decimal(repr(scale))
    q = Decimal(1).scaleb(-d)
    r = v.quantize(q, rounding=ROUND_HALF_UP)
    m = Decimal(ms.replace('−', '-').replace('+', '').replace('%', '').replace('p', '').replace(',', ''))
    diff = abs(r - m) / q
    j = '일치' if diff == 0 else ('반올림 차' if diff <= 1 else '불일치')
    return j, str(r)


rows = []


def add(sec, frag, ms, id_, f, key, src, rec='', scale=1.0, override=None, memo=''):
    r = ''
    if override:
        j = override
    elif src in (None, ''):
        j = '출처 없음'
    else:
        j, r = judge(ms, src, scale)
    if isinstance(rec, float):
        rec = f"{rec:.6g}"
    rows.append([sec, frag, ms, id_, f, key, src, rec, j, (f"사사오입 {r}; " if r else '') + memo])


MC = "(MC)\\"
T3J = MC + "T3_기술셀부상예측\\R3\\final_result_r3.json"; T3P = MC + "T3_기술셀부상예측\\R3\\preds\\FINAL_R3_t0_2021.parquet"
T3V = MC + "T3_기술셀부상예측\\VERIFY_R5\\robustness_table.md"
T1J = MC + "T1_산업대기실\\R3\\final_result_r3.json"; T1V = MC + "T1_산업대기실\\VERIFY_R5\\robustness_table.md"
N9J = MC + "N9_정부RnD과제_특허성과예측\\final_result_r6.json"
N6J = MC + "N6_정부RnD기업_첫특허전환\\final_result_r5.json"
N7J = MC + "N7_특허기업_정부RnD첫수행\\final_result_r5.json"; N7V = MC + "N7_특허기업_정부RnD첫수행\\VERIFY_R6\\robustness_table.md"
N8J = MC + "N8_특허분쟁표적레이더\\final_result_r6.json"
T4J = MC + "T4_특허포기예측\\R2\\final_result.json"; T4V = MC + "T4_특허포기예측\\VERIFY_R5\\robustness_table.md"
N4J = MC + "N4_특허행태_기업폐업조기경보\\final_result_r4.json"; N4V = MC + "N4_특허행태_기업폐업조기경보\\VERIFY_R5\\robustness_table.md"
N5V = MC + "N5_표준특허선언레이더\\VERIFY_R5\\robustness_table.md"
R12 = MC + "_shared_report\\label_origin_R10\\confirm_R12\\"
CC = MC + "_shared_report\\audit_r7\\concordance_resolution_check.json"
T5 = MC + "_shared_report\\fusion_advantage\\split34\\t5_cases.csv"
TXT = MC + "T1_산업대기실\\TXT_R9\\res\\"
M5 = MC + "_shared_report\\fusion_advantage\\map5\\"; C2 = MC + "_shared_report\\fusion_advantage\\c5_R10\\c2_results.json"
NC = MC + "_shared_report\\noncirc_n9n6_R9\\results_table.csv"
FIG7 = r"보고서_20261012\figs\fig07_t3_future_breakout_by_ksic.csv"
T1F = MC + "T1_산업대기실\\R3\\future_forecast_r3.csv"
r9, r6, r7, r8, r4, r3 = RC['N9'], RC['N6'], RC['N7'], RC['N8'], RC['N4'], RC['T3']

# ---------------- 요약 ----------------
S = '요약(9)'
add(S, '제공 50개 산업을 27개 코드로 묶어', '27', 'LBL-CW-27', CC, 'n_concordance_codes_covering_50', 27)
add(S, '34개를 서로 구분하지 못한다', '34', 'LBL-CW-27', CC, 'n_provided_ksic − ksic_alone_in_code = 50−16', 34)
add(S, '액정 표시장치의 KR 출원은 연 6.6% 줄고', '−6.6%', 'SP-CASE-26211', T5, "ksic=26211 'KR출원_연성장%(08-09→16-17)'", -6.6,
    memo='창 2008–09→2016–17(KOSIS 창과 같음). 장기 창 −9.0 미사용(mismatch 해소). 요약에는 기간 미표기(3.1.3에는 표기)')
add(S, '유기 발광 표시장치는 연 6.2% 늘었지만', '6.2%', 'SP-CASE-26212', T5, "ksic=26212 'KR출원_연성장%(08-09→16-17)'", 6.2, memo='장기 창 2.4 미사용')
S = '요약(11)'
add(S, '5개국 특허 15,141,159행', '15141159', 'DATA-META', MC + "_shared\\README.md", '§1 제목줄 "15,141,159행"', 15141159,
    memo='행=출원×KSIC 쌍. 원고도 "행"으로 적어 정합')
S = '요약(13)'
add(S, '연계표 5자리 일치는 7.1%', '7.1%', 'LBL-CW-5D', TXT + "t0_txt_agree.json", 'cw5_agree_all(16개만 지목)', 0.07095836272000927, scale=100,
    memo='분모 KR 1,087,511(미배정=불일치). R7의 7.4%(분모 1,039,164) 미사용')
add(S, '해설서 텍스트 배정은 29.4~', '29.4%', 'LBL-TXT', TXT + "t0_txt_agree.json", 'txt5_agree_all', 0.2944494354539862, scale=100)
add(S, '~40.4%', '40.4%', 'LBL-FREE', TXT + "t0b_txtfree_agree.json", 'free5_agree_all', 0.4037871800836957, scale=100,
    memo='TXT-FREE 5자리. 연계표 최빈 일치 0.4037(같은 값·다른 정의)은 원고에 없음')
add(S, '가장 강한 공개 배정도 KR 문서 라벨의 58.8%만', '58.8%', 'R12-DOCAGREE', R12 + "res_prep.json", 'doc_agree_with_prov.C-30-20.eval', 0.5876, scale=100,
    memo='평가 표본 문서 일치 사후 최고 후보(C-30-20). 같은 단락 ②의 "가장 강한 배정"(기준별 개발 1위 BEST: NPS H-sc-1 56.15%, DART H-mg-3 58.12%)과 지칭 대상이 다름')
add(S, 'DART 기준 7.3%p', '7.3%p', 'R12-DART5', R12 + "res_main.csv", '기준=DART5, PROV-BEST', 0.0731, scale=100)
add(S, '[+5.1,', '+5.1', 'R12-DART5', R12 + "res_main.csv", 'DART5, PROV-BEST_lo', 0.0505, scale=100, memo='5.05→사사오입 5.1')
add(S, '+9.7]', '+9.7', 'R12-DART5', R12 + "res_main.csv", 'DART5, PROV-BEST_hi', 0.0965, scale=100)
add(S, 'NPS 기준 −0.6%p', '−0.6%p', 'R12-NPS5', R12 + "res_main.csv", 'NPS5, PROV-BEST', -0.0062, scale=100)
add(S, '[−2.0,', '−2.0', 'R12-NPS5', R12 + "res_main.csv", 'NPS5, PROV-BEST_lo', -0.0204, scale=100)
add(S, '+1.0]', '+1.0', 'R12-NPS5', R12 + "res_main.csv", 'NPS5, PROV-BEST_hi', 0.0104, scale=100)
add(S, '해설서 단독 배정 대비 NPS +11.5%p', '+11.5%p', 'R12-T0', R12 + "res_main.csv", 'NPS5, PROV-T0', 0.1152, scale=100)
add(S, 'DART +28.7%p', '+28.7%p', 'R12-T0', R12 + "res_main.csv", 'DART5, PROV-T0', 0.2865, scale=100)
add(S, 'NPS 특허 가중 ... 공개 배정이 더 높다', '(문장)', 'R12-W', R12 + "res_main.csv", 'NPS5 PROV-BEST_w [lo,hi]', '-0.0358 [-0.076, -0.0079]', override='일치', memo='CI가 0 미만')
add(S, '대기업 구간에서는 공개 배정이 더 높다', '(문장)', 'R12-SUB', R12 + "res_sub.csv", 'NPS5 규모 >=300 PROV-BEST [lo,hi]', '-0.0398 [-0.0792, -0.0058]', override='일치', memo='81사')
add(S, 'AUC 0.002~', '0.002', 'B-P50-FREE50', TXT + "C_pool_summary.csv", 'pair=P50-FREE50 mean_d', 0.00236, memo='5/5. P50-TXT50 0.0033')
add(S, '~0.004로 작다', '0.004', 'B-P50-CW27', TXT + "C_pool_summary.csv", 'pair=P50-CW27 mean_d', 0.00413)
add(S, '87–', '87', 'CIRC-AGREE-CC', MC + "_shared_report\\label_origin_R10\\res_fam_check.json", 'links[JP][4]', 0.8696127550031832, scale=100, memo='US 87.3, JP 87.0, EP 90.5')
add(S, '–91% 일치', '91%', 'CIRC-AGREE-CC', MC + "_shared_report\\label_origin_R10\\res_fam_check.json", 'links[EP][4]', 0.904732743127205, scale=100)
S = '요약(15)'
add(S, '출원 2017–19년 한국은 ... 25개에서 RTA가 1을 넘는다', '25', 'MAP-KR-RTA', M5 + "m2_checks.json", 'kr_rta_gt1_P3', 25, memo='P3=출원 2017–19')
add(S, '기술은 강하고 수출은 약한 산업이 11개', '11', 'QUAD-COUNTS', M5 + "m4_summary.json", "quad_tech_export_P2_counts['기술 강·수출 약(괴리)']", 11, override='정의 불일치',
    memo='값은 맞음. 사분면은 P2=특허 출원 2015–17·수출 2015–17(m4_industry_quadrant.py 6행)인데 요약은 앞 문장 "출원 2017–19년"에 이어져 같은 기간으로 읽힘(표 5는 2015–17로 바르게 표기)')
add(S, '5개국 1위인 산업 수(7개)', '7', 'MAP-KR-RTA', M5 + "m2_checks.json", 'kr_leader_P3', 7)
add(S, '공개 배정으로 바꾸면 8–', '8', 'C-LAND-cweq_F', C2, 'landscape_P3.cweq_F.kr_leader', 8,
    memo='F=5개국 모두 공개 배정. H(KR 제공+해외 공개)는 9–20개 — "공개 배정으로 바꾸면"은 F와 맞음')
add(S, '–16개', '16', 'C-LAND-free_F', C2, 'landscape_P3.free_F.kr_leader', 16, memo='txt_F 12')
S = '요약(17)'
add(S, '9개 모두 skill CI 하한이 0보다 크다', '(문장)', 'fig02+각 final_result', r"보고서_20261012\figs\fig02_scorecard_9tracks.csv", 'skill_CI_lo 최소 = N5 0.007', 0.007, override='일치',
    memo='T3 0.366, T1 0.454, N8 0.243, T4 0.133, N4 0.147, N5 0.007, N9 0.362, N6 0.115, N7 0.211 — 원천 JSON과 대조')
add(S, 'T3 AUC 0.794', '0.794', 'OFF-T3-AUC', T3J, 'summary.auc', 0.7935421192942682, rec=r3['auc_p'])
add(S, '(기준선 0.642)', '0.642', 'OFF-T3-AUC', T3J, 'summary.auc_baseline', 0.6422571379591436, rec=r3['auc_p_base'])
add(S, 'skill 0.423', '0.423', 'OFF-T3-SKILL', T3J, 'summary.mean_skill', 0.422887490953899, rec=r3['skill_vs_p_base'])
add(S, '[0.366,', '0.366', 'OFF-T3-SKILL', T3J, 'ci95.skill[0]', 0.36580365189908404)
add(S, '0.473]', '0.473', 'OFF-T3-SKILL', T3J, 'ci95.skill[1]', 0.47307183171311484)
add(S, '강한 기준선 대비 0.182', '0.182', 'OFF-T3-SB3', T3V, '§3.1 SB3 행 R3 모형 skill', 0.182, memo='KR 이력 XGB 12피처(AUC 0.748)')
add(S, 'T1 AUC 0.900', '0.900', 'OFF-T1-AUC', T1J, 'model.auc_all', 0.9002187222427036,
    memo='T1 동결 예측(runs\\FROZEN_final\\o2019.parquet)은 평가 레이블 자리가 0으로 비어 재계산 불가. VERIFY_R5 원천 재구성 0.900218')
add(S, '(0.811)', '0.811', 'OFF-T1-AUC', T1J, 'baseline_strongest.auc_all', 0.8107800399723293, memo='= baselines.xgb_y_u')
add(S, 'skill 0.473', '0.473', 'OFF-T1-SKILL', T1J, 'skill.auc_all', 0.47267044268107444)
add(S, '강한 기준선 대비 0.395', '0.395', 'OFF-T1-SBB', T1V, '§3.1 SBB 행', 0.395)

# ---------------- 3.2 T3 ----------------
S = '3.2.1(205)'
add(S, '원점 2021년까지 공개된 문서', '2021', 'OFF-T3-WIN', T3J, 'origins', 2021, rec=str(r3['t0']))
add(S, '대상은 5,041셀', '5041', 'OFF-T3-N', T3J, 'by_origin.2021.model.n', 5041, rec=r3['n'])
add(S, '부상은 504셀', '504', 'OFF-T3-N', T3J, 'by_origin.2021.model.n_evt', 504, rec=r3['pos'])
add(S, 'KR 공개 2022–24년 ... 2019–21년보다 ... 상위 10%', '(창)', 'OFF-T3-WIN', T3J, 'model "E40-auditD", origins=[2021]', 't0=2021→공개 2022–24', override='일치',
    memo='VERIFY_R5 머리말 "t0=2021, KR 공개 2022–24". 상위 10% = 504/5,041')
S = '3.2.2(209)'
add(S, '모형 AUC는 0.794', '0.794', 'OFF-T3-AUC', T3J, 'summary.auc', 0.7935421192942682, rec=r3['auc_p'])
add(S, '기준선 0.642', '0.642', 'OFF-T3-AUC', T3J, 'summary.auc_baseline', 0.6422571379591436, rec=r3['auc_p_base'])
add(S, '상위 10% 셀의 36.3%', '36.3%', 'OFF-T3-TOP10', T3J, 'summary.p_at10', 0.3630952380952381, scale=100,
    rec=f"{r3['top10_p']['hits']}/{r3['top10_p']['k']}={r3['top10_p']['prec']:.4f}")
add(S, '(기준선 26.0%', '26.0%', 'OFF-T3-TOP10', T3J, 'summary.p_at10_baseline', 0.25992063492063494, scale=100,
    rec=f"quicksort {r3['top10_base_quicksort']:.4f} / stable {r3['top10_base_stable']:.4f}",
    memo='기준선 kr_mom은 동률이 많아 정렬 방식에 따라 23.8%(stable)~26.0%(quicksort). 26.0%는 기준선에 가장 유리한 값이라 모형 쪽에 보수적(VERIFY_R5 §3.4 각주)')
add(S, '무작위 10%', '10%', 'OFF-T3-TOP10', T3P, 'mean(yb)', r3['base_rate'], rec=r3['base_rate'], scale=100)
add(S, '상위 100셀에서는 50%', '50%', 'OFF-T3-TOP100', T3V, '§3.4 100셀 모형 50 (50%)', 0.50, rec=f"{r3['hit100_p']}/100", scale=100)
add(S, '(기준선 33%)', '33%', 'OFF-T3-TOP100', T3V, '§3.4 100셀 공식 기준선 33', 0.33, rec=f"{r3['hit100_p_base']}/100", scale=100)
S = '3.2.2 표6(215–218)'
add(S, '공식 skill 0.423', '0.423', 'OFF-T3-SKILL', T3J, 'summary.mean_skill', 0.422887490953899, rec=r3['skill_vs_p_base'])
add(S, '[0.366,', '0.366', 'OFF-T3-SKILL', T3J, 'ci95.skill[0]', 0.36580365189908404, memo='군집=IPC 메인그룹(ci95.cluster), 1,000회')
add(S, '0.473]', '0.473', 'OFF-T3-SKILL', T3J, 'ci95.skill[1]', 0.47307183171311484)
add(S, '보수적 skill 0.182', '0.182', 'OFF-T3-SB3', T3V, '§3.1 SB3 R3 모형 skill', 0.182)
add(S, '[0.124,', '0.124', 'OFF-T3-SB3', T3V, '§3.1 SB3 CI 하한', 0.124)
add(S, '0.243]', '0.243', 'OFF-T3-SB3', T3V, '§3.1 SB3 CI 상한', 0.243)
add(S, 'KR 이력 XGB 기준선(AUC 0.748)', '0.748', 'OFF-T3-SB3', T3V, '§3.1 SB3 R3 기준선 AUC', 0.748)
add(S, '신설 IPC 셀 제외 0.406', '0.406', 'OFF-T3-NEWIPC', T3V, '§0 IPC 2022년 이후 신설 코드 셀 제외 R3', 0.406)
add(S, '출원연도판 AUC 0.790', '0.790', 'OFF-T3APP', MC + "T3_기술셀부상예측\\R2\\final_result.json", 'summary.auc', 0.7900512626560826)
add(S, '(기준선 0.655)', '0.655', 'OFF-T3APP', MC + "T3_기술셀부상예측\\R2\\final_result.json", 'summary.auc_baseline', 0.6549923772213125,
    memo='상위 10% 37.1/37.2%는 원고에 쓰지 않음')
S = '3.2.2(220)'
add(S, '예측 상위 10%의 56.9%는 직전 3년 KR 공개가 0건', '56.9%', 'X-T3-ZERO', MC + "T3_기술셀부상예측\\R3\\FINAL_REPORT_R3.md §4", '문서값', 0.569,
    rec=r3['top10_zeroKR_share'], scale=100, memo='parquet kr_W==0 비율 재계산')
add(S, 'KR 5건 이상 셀만 봐도 AUC 0.807', '0.807', 'X-T3-KR5', T3V, '§3.2 KR ≥5건(2,195셀)', 0.807, rec=r3['krge5']['auc'], memo=f"재계산 n={r3['krge5']['n']}")
add(S, '(기준선 0.499)', '0.499', 'X-T3-KR5', T3V, '§3.2 KR ≥5건 기준선', 0.499, rec=r3['krge5']['auc_base'])
S = '3.2.3(224)'
add(S, '원점 2024년 5,328셀', '5328', 'X-T3-FUT', MC + "T3_기술셀부상예측\\R3\\future_forecast_r3_meta.json", 'n_cells', 5328, rec='5328(csv 행수)')
add(S, '상위 10%인 532셀', '532', 'X-T3-FUT', MC + "T3_기술셀부상예측\\R3\\future_forecast_r3.csv", 'sum(pred_breakout_top10)', 532, rec='532')
add(S, '비메모리 집적회로 21.9%', '21.9%', 'X-T3-FUTRATE', FIG7, 'ksic=26112 pred_rate(7/32)', 0.21875, scale=100, memo='셀 32개 중 7개 — 분모 작음')
add(S, '전기용 탄소·절연제품 19.6%', '19.6%', 'X-T3-FUTRATE', FIG7, 'ksic=28902 pred_rate(9/46)', 0.1956521739130435, scale=100)
add(S, '무인 항공기 19.5%', '19.5%', 'X-T3-FUTRATE', FIG7, 'ksic=31312 pred_rate(15/77)', 15 / 77, scale=100)
add(S, '레이더·항행기구 18.8%', '18.8%', 'X-T3-FUTRATE', FIG7, 'ksic=27211 pred_rate(13/69)', 13 / 69, scale=100)
add(S, '(평균 10%)', '10%', 'X-T3-FUT', FIG7, '532/5328', 532 / 5328, scale=100)
add(S, '532셀의 54.3%는 KR 0건', '54.3%', 'X-T3-FUT', MC + "T3_기술셀부상예측\\R3\\future_forecast_r3.csv", 'top10 중 kr_W==0 비율', 0.543233082706767, rec=0.543233082706767, scale=100)
add(S, '22셀은 IPC 2023 신설 분류', '22', 'X-T3-FUT', MC + "T3_기술셀부상예측\\R3\\FINAL_REPORT_R3.md 163행", '문서값 22', 22, rec='22(mg ∈ H10B·H10K·H10N·B64U)')
add(S, 'T3 상위 1/3(무인 항공기·전기용 탄소)', '(문장)', 'DASH-HYP', M5 + "m5_summary.json", 'tech_strong_export_weak_and_T3_top_third', '6개 목록에 28902·31312 포함', override='일치')
add(S, '제공 0.7967', '0.7967', 'T3NC-MG', MC + "T3_기술셀부상예측\\TXT_R9\\summary_mg_r9.json", 'models[model=prov].auc', 0.7967, memo='개발 원점 평균(저장값 4자리)')
add(S, '맥락 없음 0.7970', '0.7970', 'T3NC-MG', MC + "T3_기술셀부상예측\\TXT_R9\\summary_mg_r9.json", 'models[model=none].auc', 0.797, memo='저장값 0.797(4자리 저장의 끝자리 0)')

# ---------------- 3.3 T1 ----------------
S = '3.3.1(230)'
add(S, '원점 2019년 ... 2020–23년 첫 진입', '(창)', 'OFF-T1-WIN', T1J, 'origin=2019, window="2020-2023", h=4', '2019 / 2020-2023', override='일치', memo='"4년 안에" = h=4')
add(S, '특허 3건 이상', '3', 'OFF-T1-WIN', T1V + " §4B", '"가시 특허 ≥3건"', 3)
add(S, '위험쌍 1,369,514개', '1369514', 'OFF-T1-N', T1J, 'n.rows', 1369514)
add(S, '7,723개', '7723', 'OFF-T1-N', T1J, 'n.pos', 7723, memo='공개 기준 양성. 등록 기준 4,733(n.pos_reg)')
add(S, '(0.564%)', '0.564%', 'OFF-T1-N', T1J, 'n.pos/n.rows', 7723 / 1369514, scale=100)
add(S, '디자인권 법인 연결률 36.1%', '36.1%', 'LINK-T1-R3', MC + "T1_산업대기실\\R3\\ext_link_stats.csv", 'origin=2015, feature=ds_lsum, firm_link_rate', 0.3612, scale=100, override='정의 불일치',
    memo='개발 원점 2015 값(2008 31.5%, 2012 33.5%). 최종 원점 2019 값은 파일에 없음. firm_link_rate=해당 권리가 1건 이상 연결된 법인 비율(보유율 성격)')
add(S, '실용신안 31.6%', '31.6%', 'LINK-T1-R3', MC + "T1_산업대기실\\R3\\ext_link_stats.csv", 'origin=2015, feature=um_lsum, firm_link_rate', 0.3161, scale=100, override='정의 불일치',
    memo='개발 원점 2015 값(2008 39.6%, 2012 34.9%)')
add(S, '산업연관표(50/50개 산업)', '50/50', 'LINK-IO', MC + "T1_산업대기실\\R2\\ext_link_summary.csv", 'IO 행 "50/50 KSIC"', '50/50', override='일치')
S = '3.3.2(234)'
add(S, '모형 AUC는 0.900', '0.900', 'OFF-T1-AUC', T1J, 'model.auc_all', 0.9002187222427036)
add(S, '기준선 0.811', '0.811', 'OFF-T1-AUC', T1J, 'baseline_strongest.auc_all', 0.8107800399723293)
add(S, '상위 1,000쌍의 31.3%', '31.3%', 'OFF-T1-TOP1000', T1J, 'topk.model.1000.precision', 0.313, scale=100)
add(S, '(기준선 12.4%', '12.4%', 'OFF-T1-TOP1000', T1J, 'topk.strongest_auc_baseline.1000.precision', 0.124, scale=100)
add(S, '무작위 0.56%', '0.56%', 'OFF-T1-TOP1000', T1J, 'n.pos/n.rows', 7723 / 1369514, scale=100)
add(S, '상위 1%는 전체 진입의 24.0%', '24.0%', 'OFF-T1-TOP1PCT', T1V, '§3.4 1%=13,695: 1,854', 1854 / 7723, scale=100)
add(S, '(기준선 10.7%)', '10.7%', 'OFF-T1-TOP1PCT', T1V, '§3.4 공식 최강 기준선 824', 824 / 7723, scale=100)
S = '3.3.2 표7(240–243)'
add(S, '공식 skill 0.473', '0.473', 'OFF-T1-SKILL', T1J, 'skill.auc_all', 0.47267044268107444)
add(S, '[0.454,', '0.454', 'OFF-T1-SKILL', T1J, 'bootstrap_firm_cluster.skill_vs_strongest__model.auc_all[0]', 0.45362390445265505, memo='법인 군집, bootstrap_R=200')
add(S, '0.488]', '0.488', 'OFF-T1-SKILL', T1J, 'bootstrap_firm_cluster.skill_vs_strongest__model.auc_all[2]', 0.48845277547373867)
add(S, '보수적 skill 0.395', '0.395', 'OFF-T1-SBB', T1V, '§3.1 SBB 모형 skill', 0.395)
add(S, '[0.379,', '0.379', 'OFF-T1-SBB', T1V, '§3.1 SBB CI 하한', 0.379)
add(S, '0.412]', '0.412', 'OFF-T1-SBB', T1V, '§3.1 SBB CI 상한', 0.412)
add(S, 'SBB, AUC 0.835', '0.835', 'OFF-T1-SBB', T1V, '§3.1 SBB AUC', 0.835)
add(S, '국민연금·국세청 폐업일 보정 ... 양의 방향이지만 작고 CI가 겹쳤다', '(문장)', '—', T1J, 'bootstrap_firm_cluster.diff_vs_main__model_nts/_nps/_nps_nts',
    'nts auc_all +0.00057 [+0.00004,+0.00096]; nps_nts +0.00115 [+0.00052,+0.00178]; nps auc_new −0.0024 [−0.0035,−0.0012]', override='정의 불일치',
    memo='skill 주변 CI끼리는 겹치지만 짝지은 차이 CI는 NTS·NPS+NTS 전체 AUC에서 0 위(작지만 유의), NPS 변형은 신규진입자 AUC에서 음수. "양의 방향" 일률 표현은 부정확')
add(S, '개발 원점 5/5', '5/5', 'B-P50-FREE50', TXT + "C_pool_summary.csv", 'P50-FREE50 / P50-TXT50 / P50-CW27 wins', '5/5, 5/5, 5/5', override='일치')
add(S, 'AUC 0.002~0.004', '0.002~0.004', 'B-P50-*', TXT + "C_pool_summary.csv", 'mean_d', '0.00236 / 0.0033 / 0.00413', override='일치')
add(S, '(+7.3%p, NPS는 차이 없음)', '+7.3%p', 'R12-DART5', R12 + "res_main.csv", 'DART5 PROV-BEST', 0.0731, scale=100)
S = '3.3.3(249,254)'
add(S, '원점 2024년 위험쌍 1,544,097개', '1544097', 'X-T1-FUT', T1F, 'sum(at_risk)', 1544097)
add(S, '대기실 지수의 합은 9,573.6', '9573.6', 'X-T1-FUT', T1F, 'sum(waiting_room_index)', 9573.556)
add(S, '전자기 측정·분석기구 8.6%', '8.6%', 'X-T1-SHARE', T1F, 'ksic=27212 index_share', 0.08645519, scale=100)
add(S, '산업용 로봇 6.9%', '6.9%', 'X-T1-SHARE', T1F, 'ksic=29280 index_share', 0.06909036, scale=100)
add(S, '기타 의료용 기기 6.2%', '6.2%', 'X-T1-SHARE', T1F, 'ksic=27199 index_share', 0.061621, scale=100)
add(S, '기타 무선 통신장비 5.1%', '5.1%', 'X-T1-SHARE', T1F, 'ksic=26429 index_share', 0.051152, scale=100)
add(S, '광학 렌즈 3.5%', '3.5%', 'X-T1-SHARE', T1F, 'ksic=27301 index_share', 0.03465, scale=100)
add(S, '이 5개가 30.3%', '30.3%', 'X-T1-SHARE', r"보고서_20261012\figs\fig08_t1_waiting_room_index.csv", '상위 5 index_share 합', 0.302968277, scale=100)
add(S, 'KR 출원(+6.4%)', '+6.4%', 'SP-CASE-29280', T5, "ksic=29280 'KR출원_연성장%(08-09→16-17)'", 6.4, memo='장기 창 6.1 미사용')
add(S, '출하(+16.5%)', '+16.5%', 'SP-CASE-29280', T5, "ksic=29280 'KOSIS출하_연성장%'", 16.5)

# ---------------- 3.4 ----------------
S = '3.4(258)'
add(S, '과제 1,145,167건', '1145167', 'DATA-NTIS-PJ', MC + "_shared\\ext\\ntis_SOURCES_B.md", '16행', 1145167)
add(S, '성과 특허 1,094,201건', '1094201', 'DATA-NTIS-RP', MC + "_shared\\ext\\ntis_SOURCES.md", '23행', 1094201)
add(S, '제공 KR 출원의 9.34%', '9.34%', 'LINK-NTIS', MC + "_shared\\ext\\ntis_link_rates.csv", 'level=ALL link_rate(101,526/1,087,511)', 101526 / 1087511, scale=100)
S = '3.4 N9(260)'
add(S, '과제연도 2022년 과제가 2022–23년', '(창)', 'OFF-N9-WIN', N9J, 'spec_champion=C10; FINAL_REPORT_R6 12행 t=2022', 't=2022→출원 2022–23', override='일치')
add(S, '시험 표본은 75,077과제', '75077', 'OFF-N9-N', N9J, 'n', 75077, rec=r9['n'])
add(S, '양성은 5,233건', '5233', 'OFF-N9-N', N9J, 'pos', 5233, rec=r9['pos'])
add(S, '(6.97%)', '6.97%', 'OFF-N9-N', N9J, 'base_rate', 0.06970177284654422, rec=r9['base_rate'], scale=100)
add(S, 'AUC는 0.894', '0.894', 'OFF-N9-AUC', N9J, 'auc', 0.893920390168637, rec=r9['auc_p_champ'])
add(S, '기준선 0.826', '0.826', 'OFF-N9-AUC', N9J, 'auc_M0L2', 0.8256066811250664, rec=r9['auc_p_M0_L2'])
add(S, 'skill은 0.392', '0.392', 'OFF-N9-SKILL', N9J, 'skill', 0.3917220538280016, rec=r9['skill_vs_p_M0_L2'])
add(S, '[0.362,', '0.362', 'OFF-N9-SKILL', N9J, 'skill_ci95[0]', 0.362339134237114)
add(S, '0.420]', '0.420', 'OFF-N9-SKILL', N9J, 'skill_ci95[1]', 0.41952798757101567)
add(S, '강한 기준선 대비 0.356', '0.356', 'OFF-N9-STRONG', N9J, 'skill_vs_strong_L2', 0.3557319723099229, rec=r9['skill_vs_p_strong_L2'])
add(S, '[0.323,', '0.323', 'OFF-N9-STRONG', N9J, 'skill_vs_strong_L2_ci95[0]', 0.32325217529589556)
add(S, '0.384]', '0.384', 'OFF-N9-STRONG', N9J, 'skill_vs_strong_L2_ci95[1]', 0.3836591597464063)
S = '3.4 N9(265–266)'
add(S, '상위 1%(751건)', '751', 'OFF-N9-TOP1PCT', N9J, 'round(0.01×n)(k 미저장)', 751, rec=r9['top1_p_champ']['k'])
add(S, '의 78.3%', '78.3%', 'OFF-N9-TOP1PCT', N9J, 'top1', 0.7829560585885486, rec=f"{r9['top1_p_champ']['hits']}/751", scale=100)
add(S, '(기준선 54.9%', '54.9%', 'OFF-N9-TOP1PCT', N9J, 'top1_M0L2', 0.5486018641810919, rec=r9['top1_p_M0_L2']['prec'], scale=100)
add(S, '기저율 7.0%', '7.0%', 'OFF-N9-TOP1PCT', N9J, 'base_rate', 0.06970177284654422, scale=100)
add(S, '상위 10%는 39.7%', '39.7%', 'OFF-N9-TOP510', N9J, 'top10', 0.39690996270644646, rec=r9['top10_p_champ']['prec'], scale=100)
add(S, '대 31.9%', '31.9%', 'OFF-N9-TOP510', N9J, 'top10_M0L2', 0.318993074054342, rec=r9['top10_p_M0_L2']['prec'], scale=100)
add(S, '대학 0.482', '0.482', 'OFF-N9-SUB', N9J, 'subgroups.대학.skill', 0.4821292555328995)
add(S, '기업 0.257', '0.257', 'OFF-N9-SUB', N9J, 'subgroups.기업.skill', 0.25725933106515186)
add(S, '신규 수행기관 0.202', '0.202', 'OFF-N9-SUB', N9J, 'subgroups.신규수행기관.skill', 0.201940217799086)
add(S, '원판 기준선은 상위 1%에서 80.6%', '80.6%', 'OFF-N9-PILOT', N9J, 'top1_pilot_orig', 0.8055925432756325, rec=r9['top1_p_pilot_orig']['prec'], scale=100, override='정의 불일치',
    memo='80.6%는 "파일럿 설계 모형 원판"(M0_L2+M1+M2 원판, 학습 모형) 값. 같은 문장의 97건은 "강한 기준선 원판"(상위 1% 78.4%). 단수 "원판 기준선"으로 두 대상을 섞음')
add(S, '모형 93건', '93', 'OFF-N9-TOP100', N9J, 'hit100', 93, rec=r9['hit100_p_champ'])
add(S, '96', '96', 'OFF-N9-TOP100', MC + "R6_결과_20261004.md 53행", '파일럿 원판 96(문서 계산값, JSON 없음)', 96, rec=r9['hit100_p_pilot_orig'], memo='parquet 재계산으로 확인')
add(S, '97건', '97', 'OFF-N9-TOP100', MC + "R6_결과_20261004.md 53행", '강한 원판 97(문서 계산값, JSON 없음)', 97, rec=r9['hit100_p_strong_orig'], memo='parquet 재계산으로 확인')
add(S, '원판 대비 skill 0.053', '0.053', 'OFF-N9-PILOT', N9J, 'skill_vs_pilot_orig', 0.053455579315544066, rec=r9['skill_vs_p_pilot_orig'],
    memo='대상은 파일럿 원판(AUC 0.888). 강한 기준선 원판 대비는 0.259')
S = '3.4 N6(268)'
add(S, '2022–23년에 내는지', '(창)', 'OFF-N6-WIN', N6J, 'spec_champion=C11; train_t0 2012–2019', 't0=2021→2022–23', override='일치')
add(S, 'AUC는 0.747', '0.747', 'OFF-N6-AUC', N6J, 'auc', 0.7472020614306328, rec=r6['auc_p_champ'])
add(S, '(기준선 0.703)', '0.703', 'OFF-N6-AUC', N6J, 'auc_M0', 0.7027670171098742, rec=r6['auc_p_m0'])
add(S, 'skill은 0.149', '0.149', 'OFF-N6-SKILL', N6J, 'skill', 0.1494956713373372, rec=r6['skill_vs_p_m0'])
add(S, '[0.115,', '0.115', 'OFF-N6-SKILL', N6J, 'skill_ci95[0]', 0.11494606738531765)
add(S, '0.185]', '0.185', 'OFF-N6-SKILL', N6J, 'skill_ci95[1]', 0.18539980217485424)
add(S, '강화 기준선 대비 0.142', '0.142', 'OFF-N6-ENS', N6J, 'secondary_vs_M0_ensemble.skill', 0.14171797389577728, memo='강화 기준선=M0 앙상블(AUC 0.705)')
add(S, '상위 1%(123곳)', '123', 'OFF-N6-TOP1PCT', N6J, 'round(0.01×12,328)', 123, rec=r6['top1_p_champ']['k'])
add(S, '의 51.2%', '51.2%', 'OFF-N6-TOP1PCT', N6J, 'top1', 0.5121951219512195, rec=f"{r6['top1_p_champ']['hits']}/123", scale=100)
add(S, '(기준선 33.3%', '33.3%', 'OFF-N6-TOP1PCT', N6J, 'top1_M0', 0.3333333333333333, rec=f"{r6['top1_p_m0']['hits']}/123", scale=100)
add(S, '기저율 8.74%', '8.74%', 'OFF-N6-TOP1PCT', N6J, 'base_rate', 0.08744321868916288, rec=r6['base_rate'], scale=100)
S = '3.4(270)'
add(S, 'N9 +0.0001', '+0.0001', 'NC-N9-PTXT', NC, 'set=n9_y_all PROV-TXT mean_dAUC', 8.518308441310804e-05)
add(S, '2/7', '2/7', 'NC-N9-PTXT', NC, 'npos/n_origins', '2/7', override='일치')
add(S, 'N6 −0.0006', '−0.0006', 'NC-N6-PTXT', NC, 'set=n6_ntis_y_ntis PROV-TXT', -0.0005545429488649593)
add(S, '0/6', '0/6', 'NC-N6-PTXT', NC, 'npos/n_origins', '0/6', override='일치')
add(S, '우위(+0.0016,', '+0.0016', 'NC-N9-FROZEN', NC, 'set=n9_y_frozen PROV-TXT', 0.0016475301388535796)
add(S, '+0.0019)', '+0.0019', 'NC-N6-FROZEN', NC, 'set=n6_prov_y PROV-TXT', 0.0019382878193429838)

# ---------------- 3.5 ----------------
S = '3.5 박스1 N7(276)'
add(S, '2023–24년에 NTIS 과제를 처음 수행', '(창)', 'OFF-N7-WIN', N7J, 'results[0].t0=2022', 't0=2022→2023–24', override='일치')
add(S, 'AUC는 0.746', '0.746', 'OFF-N7-AUC', N7J, 'results[0].auc.champ', 0.7460584938575965, rec=r7['auc_p_champ'])
add(S, '(기준선 0.645)', '0.645', 'OFF-N7-AUC', N7J, 'results[0].auc.M0', 0.6445159723382918, rec=r7['auc_p_M0'])
add(S, '공식 skill은 0.286', '0.286', 'OFF-N7-SKILL', N7J, 'results[0].skill_vs_M0', 0.2856458057686246, rec=r7['skill_vs_p_M0'])
add(S, '[0.211,', '0.211', 'OFF-N7-SKILL', N7J, 'results[0].skill_vs_M0_ci95[0]', 0.21059617612378412)
add(S, '0.364]', '0.364', 'OFF-N7-SKILL', N7J, 'results[0].skill_vs_M0_ci95[1]', 0.3640120804312031)
add(S, '보정하면 0.251', '0.251', 'OFF-N7-ALIAS', N7V, '§1.2 skill vs M0 보정', 0.251)
add(S, '[0.180,', '0.180', 'OFF-N7-ALIAS', N7V, '§1.2 CI 하한', 0.180)
add(S, '0.324]', '0.324', 'OFF-N7-ALIAS', N7V, '§1.2 CI 상한', 0.324)
add(S, '외부 데이터를 모두 넣은 모형 대비 특허 순증분은 0.124', '0.124', 'OFF-N7-SB', N7V, '§1.2 skill vs 외부데이터 전부(ens) 보정 열', 0.124, override='정의 불일치',
    memo='0.124는 NPS 별칭 보정 후 값. 원래 키(공식 패널) 값은 0.135 [0.091, 0.179]. 원고 문장은 보정 여부를 밝히지 않음')
add(S, '상위 1%(123곳)', '123', 'OFF-N7-TOP1PCT', N7J, 'results[0].topk.champ.top1pct.k', 123, rec=r7['top1_p_champ']['k'])
add(S, '적중은 17.1%', '17.1%', 'OFF-N7-TOP1PCT', N7J, 'results[0].topk.champ.top1pct.precision', 0.17073170731707318, rec=f"{r7['top1_p_champ']['hits']}/123", scale=100)
add(S, '(기준선 10.6%', '10.6%', 'OFF-N7-TOP1PCT', N7J, 'results[0].topk.M0.top1pct.precision', 0.10569105691056911, rec=f"{r7['top1_p_M0']['hits']}/123", scale=100)
add(S, '기저율 2.60%', '2.60%', 'OFF-N7-TOP1PCT', N7J, 'results[0].base_rate', 0.026045905909164904, rec=r7['base_rate'], scale=100)
add(S, '외부 데이터만 쓴 모형도 상위 1%에서 같거나 높아', '(문장)', 'OFF-N7-SB', N7V, '§3.2 123(1%) 행: 챔피언 21/18, 외부데이터 전부 21/21', '21 대 21(공식), 18 대 21(보정)', override='일치')
add(S, '특허의 몫은 AUC와 상위 5–10% 구간', '(문장)', 'OFF-N7-SB', N7V, '§3.2 614(5%) 73 vs 66; 1,229(10%) 111 vs 93/98', '', override='일치')
add(S, '산업 피처 4개를 공식 연계표로 바꾸면 skill이 0.0102 줄었다', '0.0102', 'R7-N7', MC + "N7_특허기업_정부RnD첫수행\\ESSENTIAL_R7\\essential_summary.md", '§3 대체 KSIC 4개→연계표 Δskill', '-0.0102',
    override='일치', memo='개발 원점 4개 평균(최종시험 아님), 4/4')
S = '3.5 표8 T4(284)'
add(S, '2022-01 → 3년', '(창)', 'OFF-T4-WIN', T4J, "eval_L=['2022-01-01'], 유예 213일, DATA_END 2025-08-22", 'L=2022-01-01, 3년', override='일치',
    memo='3년+유예 213일 = 2025-08-01 < DATA_END 2025-08-22로 관측 완결')
add(S, '0.701', '0.701', 'OFF-T4-AUC', T4J, 'model.auc', 0.7010032760729219, memo='T4 최종 동결 예측 parquet을 찾지 못해 재계산 안 함(VERIFY_R5가 비트 일치 재현 보고)')
add(S, '(0.604)', '0.604', 'OFF-T4-AUC', T4J, 'baseline.auc', 0.6035757763143089)
add(S, '0.246', '0.246', 'OFF-T4-SKILL', T4J, 'skill_auc', 0.24576575783587673)
add(S, '[0.133,', '0.133', 'OFF-T4-SKILL', T4J, 'bootstrap_95ci.skill[0]', 0.1332)
add(S, '0.320]', '0.320', 'OFF-T4-SKILL', T4J, 'bootstrap_95ci.skill[1]', 0.3196)
add(S, '0.124', '0.124', 'OFF-T4-STRONG', T4V, '29행 13변수 로지스틱 skill', 0.124)
add(S, '[−0.003,', '−0.003', 'OFF-T4-STRONG', T4V, '29행 CI 하한', -0.003)
add(S, '0.208]', '0.208', 'OFF-T4-STRONG', T4V, '29행 CI 상한', 0.208)
add(S, '위험 상위 100건 중 84건', '84', 'OFF-T4-TOP100', T4V, '71행 k=100 모형 84', 84)
add(S, '(평균 19%)', '19%', 'OFF-T4-TOP100', T4J, 'base_rate', 0.18768473248996795, scale=100)
S = '3.5 표8 N8(285)'
add(S, '2022 → 2023–24', '(창)', 'OFF-N8-WIN', N8J, "t0=2022, outcome_window=['2023-01-01','2024-12-31']", '2022→2023–24', override='일치')
add(S, '0.870', '0.870', 'OFF-N8-AUC', N8J, 'champ_auc', 0.8702338559494118, rec=r8['auc_p_champ'])
add(S, '(0.780)', '0.780', 'OFF-N8-AUC', N8J, 'M0_auc', 0.7804571264578439, rec=r8['auc_p_M0'])
add(S, '0.409', '0.409', 'OFF-N8-SKILL', N8J, 'skill_vs_M0', 0.408925728460638, rec=r8['skill_vs_p_M0'])
add(S, '[0.243,', '0.243', 'OFF-N8-SKILL', N8J, 'skill_vs_M0_ci[0]', 0.24297136524642474)
add(S, '0.540]', '0.540', 'OFF-N8-SKILL', N8J, 'skill_vs_M0_ci[1]', 0.540097309715566)
add(S, '0.312', '0.312', 'OFF-N8-SB', N8J, 'skill_vs_SB', 0.31180327815646997, rec=r8['skill_vs_p_SB'])
add(S, '[0.144,', '0.144', 'OFF-N8-SB', N8J, 'skill_vs_SB_ci[0]', 0.1443316848427289)
add(S, '0.439]', '0.439', 'OFF-N8-SB', N8J, 'skill_vs_SB_ci[1]', 0.4392681700561017)
add(S, '2년 내 분쟁의 36.1%', '36.1%', 'OFF-N8-CAP1PCT', N8J, 'champ_top1pct_capture', 0.3611111111111111, scale=100,
    rec=f"{r8['top1_p_champ']['hits']}/108 (k={r8['top1_p_champ']['k']})")
add(S, '(과거 분쟁 기준선 30.6%)', '30.6%', 'OFF-N8-CAP1PCT', N8J, 'SB_top1pct_capture', 0.3055555555555556, rec=f"{r8['top1_p_SB']['hits']}/108", scale=100)
S = '3.5(287–289)'
add(S, 'N8의 양성은 108건', '108', 'OFF-N8-N', N8J, 'pos', 108, rec=r8['pos'])
add(S, '상위 1% 적중률은 1.23%', '1.23%', 'OFF-N8-PREC1PCT', N8J, 'champ_top1pct_prec', 0.012271869100062933, rec=r8['top1_p_champ']['prec'], scale=100)
add(S, '확률은 2.04배 과대', '2.04', 'OFF-N8-CALIB', MC + "R6_결과_20261004.md", '예측 합 220.3 / 108', 220.29502868652344 / 108, rec=r8['calib_ratio'])
add(S, 'N8 skill −0.0284', '−0.0284', 'R7-N8T4-FAM', MC + "N8_특허분쟁표적레이더\\ESSENTIAL_R7\\essential_summary.md", '24행 제거(5개국) Δskill', -0.0284,
    memo='개발 원점 9개 평균, 7/9 — 원고에 "개발 원점" 표기 없음')
add(S, 'T4 −0.0161', '−0.0161', 'R7-N8T4-FAM', MC + "T4_특허포기예측\\ESSENTIAL_R7\\essential_summary.md", '25행 Δskill', -0.0161, memo='개발 원점 4개, 4/4')
add(S, 'N4 AUC 0.803', '0.803', 'OFF-N4-AUC', N4J, 'results[0].auc.champ', 0.802867665495882, rec=r4['auc_p_champ'])
add(S, '(기준선 0.760)', '0.760', 'OFF-N4-AUC', N4J, 'results[0].auc.M0_xgb', 0.760477643184264, rec=r4['auc_p_M0'])
add(S, 'skill 0.177', '0.177', 'OFF-N4-SKILL', N4J, 'results[0].skill_vs_M0', 0.1769773096556013, rec=r4['skill_vs_p_M0'])
add(S, '특허 순증분은 0.052', '0.052', 'OFF-N4-PATINC', N4V, '28행 외부데이터 전부 앙상블 대비', 0.052, memo='공식(원래 키) 패널 값. 보정 패널 0.055')

# ---------------- 4.2 ----------------
S = '4.2 표11(388)'
add(S, '후보 셀 100개 중 실제 부상 50개', '50', 'OFF-T3-TOP100', T3V, '§3.4 100셀 50', 50, rec=r3['hit100_p'])
add(S, '(모멘텀 기준선 33개)', '33', 'OFF-T3-TOP100', T3V, '§3.4 100셀 33', 33, rec=r3['hit100_p_base'])
add(S, '보수적 skill 0.182', '0.182', 'OFF-T3-SB3', T3V, '§3.1 SB3', 0.182)
S = '4.2 표11(389)'
add(S, '진입 후보 1,000쌍 중 313쌍', '313', 'OFF-T1-TOP1000', T1J, 'topk.model.1000.hits', 313)
add(S, '(기준선 124쌍', '124', 'OFF-T1-TOP1000', T1J, 'topk.strongest_auc_baseline.1000.hits', 124)
add(S, '무작위 약 6쌍', '6', 'OFF-T1-N', T1J, '1000×n.pos/n.rows', 1000 * 7723 / 1369514, memo='5.64쌍 → 약 6')
S = '4.2 표11(390)'
add(S, '상위 10% 점검으로 성과 과제의 56.9% 포착', '56.9%', 'OFF-N9-TOP510', N9J, 'recall_top10', 0.5694630231224919, rec=r9['top10_p_champ']['capture'], scale=100)
add(S, '(기준선 45.8%)', '45.8%', 'OFF-N9-TOP510', N9J, 'recall_top10_M0L2', 0.45767246321421745, rec=r9['top10_p_M0_L2']['capture'], scale=100)
add(S, '상위 1%는 1년 더 기다린 기준선과 같거나 낮음', '(문장)', 'OFF-N9-PILOT', N9J, 'top1 0.783 vs top1_strong_orig 0.784 / top1_pilot_orig 0.806', '', override='정의 불일치',
    memo='"1년 더 기다린"이 아니라 "이력 정보를 1년 더 쓰는(≤t−1)" 원판. 비교 대상에 파일럿 원판 모형이 섞임')
S = '4.2 표11(391)'
add(S, '시험 특허 317,840건', '317840', 'OFF-N8-N', N8J, 'n', 317840, rec=r8['n'])
add(S, '의 1/100', '3178', 'OFF-N8-CAP1PCT', N8J, 'k_top1pct', 3178, rec=r8['top1_p_champ']['k'], memo='317,840/100=3,178.4 → 3,178건(원고는 건수 미표기)')
add(S, '분쟁 108건 중 39건', '39', 'OFF-N8-CAP1PCT', N8J, 'champ_top1pct_hits', 39, rec=r8['top1_p_champ']['hits'])
add(S, '(36.1%)', '36.1%', 'OFF-N8-CAP1PCT', N8J, 'champ_top1pct_capture', 0.3611111111111111, scale=100)
add(S, '단순 기준선 3.7%', '3.7%', 'OFF-N8-CAP1PCT', N8J, 'M0_top1pct_capture', 0.037037037037037035, rec=f"{r8['top1_p_M0']['hits']}/108", scale=100, override='정의 불일치',
    memo='M0는 IDEAS_R5 정의 14개 피처 XGB(FINAL_REPORT_R6 22행)인 공식 기준선이지 "단순" 기준선이 아님. 3.5 표 8은 같은 M0를 "기준선(0.780)"으로 씀')
add(S, '과거 분쟁 기준선(30.6%)', '30.6%', 'OFF-N8-CAP1PCT', N8J, 'SB_top1pct_capture', 0.3055555555555556, scale=100)
S = '4.2 표11(392)'
add(S, '상위 100건 중 84건 3년 내 포기', '84', 'OFF-T4-TOP100', T4V, '71행', 84)
add(S, '정부 R&D 기원 특허 5년 내 포기 14.2%', '14.2%', 'NTIS-T3-SUM', MC + "_shared_report\\ntis_t3_cohort_lapse_transfer.csv", 'group=정부R&D 기원(전체) lapse5', 0.1423, scale=100,
    memo='등록 2007–2016 코호트(원고 4.2는 코호트 미표기)')
add(S, '(비기원 21.0%)', '21.0%', 'NTIS-T3-SUM', MC + "_shared_report\\ntis_t3_cohort_lapse_transfer.csv", 'group=정부R&D 기원 아님 lapse5', 0.2097, scale=100)
add(S, '보수적 skill 0.124', '0.124', 'OFF-T4-STRONG', T4V, '29행', 0.124)
S = '4.2 표11(393,396)'
add(S, "'기술 강·수출 약' 11개 산업", '11', 'QUAD-COUNTS', M5 + "m4_summary.json", 'quad_tech_export_P2_counts 기술 강·수출 약', 11)
add(S, 'T3 상위 1/3인 6개로 압축', '6', 'DASH-HYP', M5 + "m5_summary.json", 'len(tech_strong_export_weak_and_T3_top_third)', 6)
add(S, '순위상관은 −0.611~', '−0.611', 'DASH-RANGE', M5 + "m5_summary.json", 'track_pair_spearman_range[0]', -0.611)
add(S, '~+0.507', '+0.507', 'DASH-RANGE', M5 + "m5_summary.json", 'track_pair_spearman_range[1]', 0.507,
    memo='m5 대시보드의 T4 열은 R3 X26 미래 예측(공식 R2 아님)')

# ---------------- 4.4 ----------------
S = '4.4(413)'
add(S, 'NPS 특허 가중(−3.6%p)', '−3.6%p', 'R12-W', R12 + "res_main.csv", 'NPS5 PROV-BEST_w', -0.0358, scale=100)
add(S, '가입자 300명 이상 기업(−4.0%p)', '−4.0%p', 'R12-SUB', R12 + "res_sub.csv", 'NPS5 규모 >=300 PROV-BEST', -0.0398, scale=100)
add(S, '그룹 안 5자리 선택(−21.7%p)', '−21.7%p', 'R12-WITHIN', R12 + "res_within.csv", 'NPS5 PROV-BEST', -0.2168, scale=100)
add(S, 'DART +7.3%p만 남고', '+7.3%p', 'R12-DART5', R12 + "res_main.csv", 'DART5 PROV-BEST', 0.0731, scale=100)
add(S, '근거 B는 AUC 0.002~0.004', '0.002~0.004', 'B-P50-*', TXT + "C_pool_summary.csv", 'mean_d', '0.00236~0.00413', override='일치')
S = '4.4(415)'
add(S, 'T3 시험 구간(2022–24)', '2022–24', 'OFF-T3-WIN', T3J, 'origins=[2021] → 공개 2022–24', '2022–24', override='일치')
add(S, 'T4·N5는 강화 기준선 대비 CI가 0에 닿는다', '(문장)', 'OFF-T4-STRONG / OFF-N5-STRONG', N5V + ' ; ' + T4V, 'N5 29행 skill 0.0002 [−0.072, 0.070]; T4 0.124 [−0.003, 0.208]', '',
    override='정의 불일치', memo='T4는 하한이 0에 닿는 수준(−0.003)이지만 N5는 점추정 0.000으로 강한 기준선 대비 이득이 없음. "닿는다"는 N5를 과대 표현(2.3 끝 문장도 같음)')
add(S, 'N9 상위 1%와 N8 맨 위 목록은 강한 기준선 수준', '(문장)', 'OFF-N9-PILOT / OFF-N8', N9J + ' ; ' + N8J,
    'N9 top1 0.783 vs strong_orig 0.784·pilot_orig 0.806; N8 hit100 18 vs SB 17, top1% 39 vs 33', '', override='일치')
S = '4.4(416)'
add(S, '국민연금 상호 키 보정 후 N4는 0.140', '0.140', 'OFF-N4-ALIAS', N4V, '84행 보정 skill', 0.140)
add(S, 'N7은 0.251', '0.251', 'OFF-N7-ALIAS', N7V, '§1.2', 0.251)

with open(W + "audit_numbers.csv", 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['절', '원고 문장 조각', '원고 값', 'id', '원천 파일', '키', '원천 값', '재계산 값', '판정', '메모'])
    w.writerows(rows)
c = Counter(r[8] for r in rows)
print(len(rows), dict(c))
for r in rows:
    if r[8] != '일치':
        print(r[0], '|', r[2], '|', r[8], '|', r[9][:110])

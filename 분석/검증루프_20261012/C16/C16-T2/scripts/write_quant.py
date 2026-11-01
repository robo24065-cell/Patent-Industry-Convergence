# -*- coding: utf-8 -*-
import json, csv, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
q = json.load(open(os.path.join(H, "cache", "q1_out.json"), encoding="utf-8"))
b = json.load(open(os.path.join(H, "cache", "q1b_out.json"), encoding="utf-8"))
rows = [
 ["Q1a", "N4", "N4-08", "DART 재무 패널 기업 중 국세청 registered 연결", q["a_matched_to_nts_registered"], q["a_dart_bz_firms"], round(q["a_matched_to_nts_registered"]/q["a_dart_bz_firms"], 4), "사전등록", r"cache\q1_out.json"],
 ["Q1a", "N4", "N4-08", "그중 국세청 폐업일 보유(2016–2026)", q["a_closed_any"], q["a_matched_to_nts_registered"], round(q["a_closed_share"], 4), "사전등록", "연도별 " + json.dumps(q["a_closed_by_year"], ensure_ascii=False)],
 ["Q1a", "N4", "N4-08", "2026년 전 폐업 기업 수(DART 패널 안)", q["a_closed_lt2026"], "", "", "사전등록", "판정 문턱 20곳 이상 충족"],
 ["Q1a", "N4", "N4-08", "DART 패널 폐업 비율(폐업일 2016–2025)", "", "", round(q["a_closed_2016_2025_share"], 4), "사전등록", ""],
 ["Q1b", "N4", "N4-08", "국세청 대상 모집단(registered) 폐업 비율(2016–2025)", "", q["nts_registered_bz"], round(q["b_univ_closed_2016_2025_share"], 4), "사전등록", "tier1(제공범위 등록 보유) 전체 기간 " + str(round(q["b_tier1_closed_share"], 4))],
 ["Q1b", "N4", "N4-08", "비율 비(DART 패널/모집단)", "", "", round(q["ratio_dart_vs_univ_2016_2025"], 4), "사전등록", "문턱 0.25 미만 → 규칙상 'B 의심'"],
 ["Q1c", "N4", "N4-08", "N4 시험 패널(t0=2023) dart_has=1 비율(영향 표본)", q["c_by_dart_has"]["1.0"]["n"], q["c_test_n"], round(q["c_dart_has_rate"], 4), "사전등록", "y 결합 24,840/24,840"],
 ["Q1c", "N4", "N4-08", "dart_has=1 층 폐업률", q["c_by_dart_has"]["1.0"]["pos"], q["c_by_dart_has"]["1.0"]["n"], round(q["c_by_dart_has"]["1.0"]["rate"], 4), "사전등록", ""],
 ["Q1c", "N4", "N4-08", "dart_has=0 층 폐업률", q["c_by_dart_has"]["0.0"]["pos"], q["c_by_dart_has"]["0.0"]["n"], round(q["c_by_dart_has"]["0.0"]["rate"], 4), "사전등록", ""],
 ["Q1d", "N4", "N4-08", "규모(NPS 고용 구간) 표준화 폐업률 비 dart_has=1/0", "", "", round(b["ratio_size_std"], 4), "사후(PREREG 밖)", "dart_has=0 표준화 " + str(round(b["dart0_rate_size_std"], 4)) + " 대 dart_has=1 " + str(round(b["dart1_rate"], 4)) + "; 50–299명 0.0066 대 0.0092, 300명+ 0.008 대 0.012"],
 # 인용(새로 재지 않음)
 ["인용", "T1", "T1-01", "원점 연말 등록분만으로는 3건 미만인 평가 법인", 3260, 28798, 0.1132, "인용", "FINDINGS 350(C14-T3), Δskill +0.0028"],
 ["인용", "T1", "T1-02", "신규진입자 쌍(위험쌍 대비)", 64999, 1369514, 0.0475, "인용", "T1 R3 FINAL_REPORT_R3.md:47; 양성 1,528/7,723=0.198"],
 ["인용", "T1", "T1-04", "선행출원 제외 쌍", 7765, "", "", "인용", "FINAL_REPORT_R3.md:44, FINDINGS 65"],
 ["인용", "T1", "T1-05", "RC:001 제외 법인 코드(모집단 대비 2.1%)", 2465, "", 0.021, "인용", "FINDINGS 251(C10-T1)"],
 ["인용", "N4", "N4-01", "as-of 미확인(2026 목록에만 의존) 시험 법인", 4039, 24840, 0.1626, "인용", r"C05\C05-T2\rescore_strata.csv S2"],
 ["인용", "N7", "N7-01", "as-of 미확인 시험 법인", 3639, 12286, 0.2962, "인용", r"C05\C05-T2\rescore_strata.csv S2; 확인 층 skill 0.184"],
 ["인용", "N6", "N6-02", "시험 위험집합 중 미연결(결과 구조적 0) 기업", 12328 - 9709, 12328, round((12328 - 9709) / 12328, 4), "인용", "N6 FINAL_REPORT_R5.md:26,118; 연결 기업만 skill 0.147"],
 ["인용", "N6", "N6-02", "위험집합 '첫 특허' 이전 기업 혼입", 119, 12328, round(119 / 12328, 4), "인용", "FINDINGS 157(C06-T3)"],
]
with open(os.path.join(H, "quant.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["측정ID", "트랙", "요소ID", "항목", "분자", "분모", "비율·값", "구분", "출처·비고"]); w.writerows(rows)
print(len(rows))

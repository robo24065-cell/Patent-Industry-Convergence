# C15-T4 s2: 고정 목록(cache\candidates_frozen.csv) 137단위에 1차 판정을 붙여 claim_map_ext.csv를 만든다.
# 판정 근거(값)는 원 결과 파일에서 직접 확인한 것을 note에 적는다(확인 스크립트: s3_evidence.py, cache\evidence.json).
import csv, json, math, sys, hashlib
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C15\C15-T4"
fr = list(csv.DictReader(open(W + r"\cache\candidates_frozen.csv", encoding="utf-8-sig")))
assert hashlib.sha256(open(W + r"\cache\candidates_frozen.csv", "rb").read()).hexdigest().startswith("10693633")
NA = "대상 아님"; S = "지지"; G = "단서 누락"; X = "과장"
# uid: (verdict, track, unit, findings_lines, result_files, errata_ref, fix_id, priority, note)
J = {
"U001": (S, "정의", "연계 다리", "", "원고 표 3(84–97행)", "", "", "", "변경 줄. 다리 ①②③ 구성이 표 3과 같음"),
"U002": (S, "정의", "설계 원칙", "", "", "", "", "", "변경 줄. 뒤에 ①–④ 네 개"),
"U003": (S, "정의", "원칙 ①", "339", r"C14\C14-T2\diff_audit.csv", "E-C14T2-02(반영됨)", "", "", "변경 줄. ERRATA 전체판 원안과 문자 일치. T1·T3·T4·N8·N4·N5·N7 모집단과 N6·N9 결과가 제공 데이터 → '모집단이나 결과' 성립. 부수: 괄호 '등록특허 656,360건'은 실용신안 41,318건 포함(U003-b)"),
"U004": (S, "정의", "원칙 ② 예외", "309,349", r"C14\C14-T3\data_check.txt", "E-C14T3-02(반영됨)", "", "", "변경 줄. 원안 일치. 다른 비시점 값(IPC 소급·CA 2026 연결)은 4.4-3·4.4-4에 단서 → 지지"),
"U005": (S, "정의", "원칙 ③", "173", r"C07\C07-T1\data_check.txt", "", "", "", "변경 줄(문구 불변). N8 로지스틱 보조 챔피언의 KOSIS 피처는 공식 XGB 무관"),
"U006": (S, "정의", "원칙 ④", "", "", "", "", "", "변경 줄(문구 불변)"),
"U007": (X, "T1·T3", "잠긴 최종시험", "", r"(MC)\R2_결과_20260928.md 1.1표(30–33행)·86행; (MC)\_shared\SPLITS.md 18행; (MC)\_shared\PROTOCOL_R3.md 16–20행; (MC)\R3_결과_20260929.md 196–214행; ERRATA 354행", "", "F1", 1,
         "모순 경계. '처음부터 잠가 정답을 열어 보지 않았고'는 라운드 안에서만 성립. T1 공식 창(c=2019→2020–23 진입)은 2라운드 최종시험(c=2017→2018–22, skill 0.418)이 이미 채점한 2020–22와 겹치고, "
         "T3 공식 창(t0=2021→KR 공개 2022–24)은 2라운드 최종시험(t0=2019·2020→출원 2020–22·2021–23, 0.392)·1라운드 홀드아웃(t0=2018→2019–21, 0.410)과 결과 문서가 대부분 겹친다. "
         "그 사이 결과 정의(출원→공개)·원점·창 길이를 바꿨다. PROTOCOL_R3의 '아직 아무도 보지 않은 구간'도 같은 문제. R3_결과 3.3 원칙('이전 라운드 공식값은 다른 정의의 과거 최종시험으로 병기')과 달리 원고 본문에는 병기 없음(표 6 '출원연도판' 행은 v6 지면 조정 때 삭제, 부록 OFF-T3APP만 남음). 공식 성적 값 자체는 동결 모형의 첫 채점이라 맞음"),
"U008": (NA, "", "", "", "", "", "", "", "AUC 정의"),
"U009": (S, "공통", "보수적 skill", "", "원고 표 6–8", "", "", "", ""),
"U010": (S, "T3", "해외 문서 제거", "263", r"(MC)\_shared_report\fusion_advantage\c5_R10\c5_t3_summary.json", "", "", "", "개발 +0.0339 [+0.0285, +0.0390]; 공개 배정으로 채우면 23–29% 보존 → '일부 회복'"),
"U011": (NA, "", "", "", "", "", "", "", "절 머리"),
"U012": (S, "T3", "순환 효과", "", r"(MC)\T3_기술셀부상예측\TXT_R9\summary_mg_r9.json; (MC)\T3_기술셀부상예측\ESSENTIAL_R7\essential_table_pub.csv", "", "", "", "라벨 무관 메인그룹 결과에서 연계표 0.7950 대 제공 0.7967(차 0.0017) vs KSIC 셀 결과 −0.199 → 대부분 순환"),
"U013": (S, "필수성", "사전 등록·사후 표시", "", "원고 321·323·324·326행 표기", "", "", "", "3.6의 사후·판정 외 결과는 '(사후)'·'판정 외'·'참고'로 표시. 검증루프 추가 시험(C10-T2·C14-T1 등)은 각 PREREG"),
"U014": (S, "근거A", "확인 시험 경위", "139", r"(MC)\_shared_report\label_origin_R10\confirm_R12\PREREG.txt", "", "", "", ""),
"U015": (NA, "", "", "", "", "", "", "", "변경 줄의 글머리 제목"),
"U016": (S, "트렌드", "WIPO 보정", "", r"(MC)\_shared_report\fusion_advantage\map5\m2_checks.json, m7_summary.json", "", "", "", "변경 줄(문구 불변)"),
"U017": (S, "트렌드", "지형도 원계열", "2", "", "E-C01-01(반영됨)", "", "", "변경 줄(문구 불변)"),
"U018": (S, "T1·T3", "빈티지·스냅샷", "309,338,347,349", r"C10\C10-T1\sens_results.csv; C12\C12-T3\b_sens.csv; C14\C14-T3\sens_results.csv", "E-C14T2-01·E-C14T3-01(반영됨)", "", "", "변경 줄. 원안과 문자 일치. 하락 최대 T1 산업연관표 −0.00221(T3 생산지수는 최종 +0.0082·개발 −0.0010), 권리목록 S3 −0.0069. DART(검증 기준)·CA 2026 연결은 4.4-4 단서"),
"U019": (G, "데이터", "공식 통계 대비 비율", "45(확인); 6·9(참고)", r"C02\C02-T2\official_vs_provided.csv", "E-C02T2-04(반영됨·축약)", "F2", 2,
         "변경 줄. 원천 CSV 직접: KR ratio_pat 2012–23 0.253–0.263(2024 0.209), CN ratio_t3year 2018–22 0.201–0.207. 그러나 CN 발명 공개연도 비율 2012–14 0.105–0.164(2015–22 0.215–0.230), CN 등록일 기준 2021·22·24 0.131–0.143. "
         "ERRATA 원안의 기간 조건('KR 2012–23년', 'CN 2018–22년')이 축약 반영 때 빠져 '연도별 건수 안정'이 전 기간으로 읽힘. 4.4는 KR 2024 등록·CN 2021 IPC만 다룸"),
"U020": (G, "T1", "그림 3 출처", "346,347,349", r"C14\C14-T3\summary.txt 35행; (MC)\T1_산업대기실\R3\FINAL_REPORT_R3.md 94행", "", "F4", 2,
         "캡션 출처 '제공 데이터와 표 3의 외부 데이터'. T1 피처 1–4위(t_top3·ot_top3·t_lsum·t_rrec2_max)를 포함한 범위밖 의존 42열(gain 48.8%)이 사업자 권리목록(KIPRIS Plus, 2026)에서 오는데 표 3·각주 ¹에 없음(표 13에만 있음)"),
"U021": (NA, "", "", "", "", "", "", "", "계산 기술"),
"U022": (S, "트렌드", "격차", "", r"(MC)\_shared_report\fusion_advantage\map5\m2_ksic_profile.csv", "", "", "", "0.11·0.15 다음은 0.32"),
"U023": (S, "트렌드", "선행 관계", "53", r"(MC)\_shared_report\fusion_advantage\map5\m3_summary.json", "", "", "", ""),
"U024": (S, "트렌드", "그림 4 캡션", "2", "", "E-C01-01(반영됨)", "", "", ""),
"U025": (S, "트렌드", "WIPO 투영 ρ", "3", r"C01\C01-T1\verify_1\v1_results.json", "", "", "", "KR 0.449 최저(3년 창 7개 모두 최저). 부트스트랩 최저 확률 71%는 '점추정 서술' 권고와 맞음"),
"U026": (S, "트렌드", "IO 단위 RCA", "71", r"C03\C03-T2\cache\s2c_io_structure.csv", "E-C03-T2-01(반영됨)", "", "", ""),
"U027": (S, "트렌드", "영상 기기(경계)", "73", r"C03\C03-T2\quadrant_sensitivity.csv", "E-C03-T2-03(반영됨)", "", "", "11개 변형 중 6개 WW"),
"U028": (NA, "", "", "", "", "", "", "", "표 주 기술"),
"U029": (S, "트렌드", "사분면 지속", "69", r"(MC)\_shared_report\fusion_advantage\map5\m4_summary.json", "", "", "", "[사후] 표시 있음"),
"U030": (S, "트렌드", "코드 안 사분면", "", r"(MC)\_shared_report\fusion_advantage\map5\m4_summary.json", "", "", "", "QUAD-CWMIX 6/11. 무역 정의 변형별 재집계는 없음(참고)"),
"U031": (G, "트렌드", "코드 안 분산", "(원천 직접)", r"(MC)\_shared_report\fusion_advantage\split34\t1_variance_decomposition.csv; C13\C13-T3\verify_1\refute_note.txt(K041)", "", "F3", 3,
         "원천 CSV 직접: 그룹 안 분산 비중 KR 0.561·US 0.551·JP 0.596·EP 0.563 대 CN 0.411·5개국 합 0.291. '세계 추세는 읽히지만 한국 지표는 그렇지 않다'는 숫자는 맞으나 US·JP·EP도 같아 한국만의 차이가 아니고, '세계 추세'는 CN 비중이 큰 합계"),
"U032": (NA, "", "", "", "", "", "", "", "T3 정의(E-C04T3-01 반영본)"),
"U033": (NA, "", "", "", "", "", "", "", "입력 기술"),
"U034": (S, "T3", "공식 AUC", "99", r"(MC)\T3_기술셀부상예측\R3\final_result_r3.json", "", "", "", ""),
"U035": (S, "T1", "문제", "", "", "", "", "", "변경 줄(문구 불변)"),
"U036": (S, "T1", "단위", "", "", "", "", "", "변경 줄(문구 불변)"),
"U037": (S, "T1", "모집단 정의", "350", r"C14\C14-T3\cache\s5_out.json", "E-C14T3-03(반영됨)", "", "", "변경 줄. 원안 일치"),
"U038": (S, "T1", "위험쌍 수", "", r"(MC)\T1_산업대기실\R3\final_result_r3.json", "", "", "", "변경 줄(문구 불변)"),
"U039": (G, "T1", "외부 데이터 목록", "346,347,349", r"C14\C14-T3\summary.txt 8·35행; (MC)\T1_산업대기실\R3\FINAL_REPORT_R3.md 94·140행", "", "F4", 2,
         "변경 줄. C14 새 사실과 불일치: T1 예측 피처 중 범위밖 의존 42열(gain 48.8%, 1–4위 피처 포함)이 2026 사업자 권리목록(KIPRIS Plus)에서 오는데 외부 데이터 목록에는 디자인권·실용신안·물가·산업연관표만 있음(143행에는 'T1 권리목록'이 나옴)"),
"U040": (S, "T1", "공식 AUC", "243", r"(MC)\T1_산업대기실\R3\final_result_r3.json", "", "", "", ""),
"U041": (S, "T1", "미래 지수 해석", "", r"(MC)\T1_산업대기실\VERIFY_R5\robustness_table.md", "E-C13T3-K057(미반영·선택)", "", "", "미래 원점 확률 보정은 검증 불가 → '미검증' 맞음. 공식 시험 약 1.5배 과대는 기존 제안 K057"),
"U042": (G, "N9", "AUC 우위", "130(작업자); 178(확인)", r"(MC)\N9_정부RnD과제_특허성과예측\final_result_r6.json", "E-C05T3-02(반영됨·축약)", "F5", 3,
         "원천 json 직접: 모형 AUC 0.8939 > pilot_orig 0.8879·strong_orig 0.8568이지만 원판 챔피언(secondary_orig_champion, 정보 1년 더) 0.8993 > 0.8939. E-C05T3-02 [확인] 원안의 '(원판 챔피언: AUC 0.899 …)'가 축약 반영 때 빠져 '우위는 순위 전반(AUC)'이 일반화로 읽힘"),
"U043": (S, "N6", "표기 오류 영향", "32", r"C02\C02-T1\verify_0\verify_note.txt", "E-C02T1-03(반영됨)", "", "", "개발 원점 양성률 차 ≤0.09%p"),
"U044": (S, "N9·N6", "분야 무관 결과", "", r"(MC)\_shared_report\noncirc_n9n6_R9\results_table.csv", "", "", "", ""),
"U045": (S, "N7", "특허 순증분", "13", r"(MC)\N7_특허기업_정부RnD첫수행\VERIFY_R6\robustness_table.md 111행", "E-C01-09(반영됨)", "", "", ""),
"U046": (S, "N7", "상위 k 몫", "", r"(MC)\N7_특허기업_정부RnD첫수행\VERIFY_R6\robustness_table.md 3.2(118–128행)", "", "", "", "123건 21 대 21(보정 18 대 21), 5% 73 대 66, 10% 111 대 93–98"),
"U047": (S, "필수성", "같은 단위 없음", "", "원고 1.1", "", "", "", "바로 뒤 일치율로 한정"),
"U048": (S, "필수성", "공개 재배정 일치", "", "부록 LBL-*·R12-DOCAGREE", "", "", "", ""),
"U049": (S, "근거A", "격차 축소", "", "원고 표 9", "", "", "", "TXT-FREE 대비 +11.5/+28.7 → FREE-IPC 기본 대비 +1.0/+9.3"),
"U050": (S, "근거A", "공개 쪽 유리", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\PREREG.txt", "", "", "", ""),
"U051": (NA, "", "", "", "", "", "", "", "글머리 제목"),
"U052": (NA, "", "", "", "", "", "", "", "판정 규칙 정의(R12 PREREG와 같음)"),
"U053": (S, "근거A", "재현 점검", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_prep.json; res_tests.json", "", "", "", ""),
"U054": (S, "근거A", "그림 8 캡션", "", "", "", "", "", "NPS·DART 출처 명시"),
"U055": (NA, "", "", "", "", "", "", "", "정의"),
"U056": (S, "근거A", "판정 부분", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_tests.json", "", "", "", ""),
"U057": (S, "근거A", "사후 최강", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_main.csv", "", "", "", "0.6286−0.5617"),
"U058": (S, "근거A", "IPC 최빈표로 메워짐", "", "원고 표 9", "", "", "", "NPS 91%, DART 68%"),
"U059": (NA, "", "", "", "", "", "", "", "글머리 제목"),
"U060": (S, "근거A", "특허 가중", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_main.csv", "", "", "", ""),
"U061": (S, "근거A", "구간", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_sub.csv", "", "", "", ""),
"U062": (S, "근거A", "나머지 구간", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_sub.csv", "", "", "", ""),
"U063": (NA, "", "", "", "", "", "", "", "글머리 제목"),
"U064": (S, "근거A", "3자리", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_main.csv", "", "", "", "셋 다 양수, NPS3 하한>0"),
"U065": (NA, "", "", "", "", "", "", "", "글머리 제목"),
"U066": (S, "근거A", "출원인 이력 배정", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_app.csv", "", "", "", "판정 외 참고. NPS는 −0.6→−0.7(줄지 않음), DART +2.7은 CI 0 포함 — 표현 경미"),
"U067": (NA, "", "", "", "", "", "", "", "글머리 제목"),
"U068": (NA, "", "", "", "", "", "", "", "조건문"),
"U069": (S, "근거A", "순환 점검", "145", r"(MC)\_shared_report\label_origin_R10\res_p1.csv", "", "", "", ""),
"U070": (S, "근거A", "발명자 문서·다업종", "", r"(MC)\_shared_report\label_origin_R10\res_posthoc2.json; res_p3.csv", "", "", "", "(사후) 표시 있음"),
"U071": (S, "근거A", "정리", "", "", "", "", "", "DART 2026 스냅샷 단서는 4.4-4"),
"U072": (S, "근거A", "정리", "", "", "", "", "", ""),
"U073": (S, "근거B", "순환 없음", "", "", "", "", "", "결과 정의가 라벨과 무관"),
"U074": (S, "근거B", "AUC", "254", r"(MC)\T1_산업대기실\TXT_R9\res\C_summary.csv", "", "", "", "변경 줄(문구 불변)"),
"U075": (S, "근거B", "이원 군집 구조명", "344,345", r"C14\C14-T1\se_compare.csv; res\verdict.json", "E-C14T1-01(반영됨)", "", "", "변경 줄. 원안 일치. 법인×대상 업종 CGM [−0.00016, +0.00489]"),
"U076": (S, "근거B", "크기 작음", "255", "", "", "", "", ""),
"U077": (NA, "", "", "", "", "", "", "", "글머리 제목"),
"U078": (S, "근거C", "KR·해외 재현 비슷", "", "원고 표 2 주", "", "", "", "방법 차이 단서는 표 2 주(73행)"),
"U079": (S, "근거C", "KR 라벨 체계 몫", "340", r"(MC)\_shared_report\label_origin_R10\foreign_public_R11\res_main.csv", "", "", "", "설명 몫 NPS 51%·DART 65% — '대부분'은 NPS에서 경계"),
"U080": (S, "근거A", "표 10", "", "", "", "", "", ""),
"U081": (S, "근거A", "표 10", "", "", "", "", "", ""),
"U082": (S, "근거A", "표 10", "", "", "", "", "", "DART +5.3은 Δ 칸에 있음"),
"U083": (S, "N9·N6", "표 10", "", "", "", "", "", ""),
"U084": (S, "근거C", "표 10", "262", "", "", "", "", ""),
"U085": (S, "근거C", "표 10", "", "", "", "", "", ""),
"U086": (G, "필수성", "종합", "277; 171(참고)", r"원고 표 10(355행); C11\C11-T2\res\s5_summary_mg__pure.json", "", "F6", 3,
         "'예측 피처(T3·N9·N6) … 같고'는 표 10 355행 '같다(경계)'와 강도가 다름(대안 B +0.0026 [+0.0004, +0.0046] 7/9). C13이 336행에 '(경계)'를 넣은 것과 같은 유형. 보수 방향"),
"U087": (NA, "", "", "", "", "", "", "", "부록 지표명 '일치'"),
"U088": (NA, "", "", "", "", "", "", "", "부록 지표명 '일치'"),
"U089": (NA, "", "", "", "", "", "", "", "부록 지표명 '일치'"),
"U090": (S, "필수성", "후보 최고", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_prep.json", "", "", "", ""),
"U091": (S, "근거A", "R12 판정", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_tests.json", "", "", "", ""),
"U092": (S, "근거A", "BEST", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_tests.json", "", "", "", ""),
"U093": (S, "근거A", "재현 점검", "", r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_prep.json", "", "", "", ""),
"U094": (NA, "", "", "", "", "", "", "", "부록 개수"),
"U095": (NA, "", "", "", "", "", "", "", "부록 개수"),
"U096": (S, "트렌드", "동률 규칙 의존", "279", r"C11\C11-T2\verify_5\v5_frac.json(desc_free_F 0.4727); refute_result.txt 9행", "", "A1", 3,
         "0.473 = 표 동률만 반대 처리(v5_frac 0.4727) 맞음. 다만 본문 168행 '최대 0.48'은 배정까지 반대 처리한 0.4755라 부록에 출처값이 없음 → 부록 동기화 제안(판정에는 넣지 않음)"),
"U097": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U098": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U099": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U100": (S, "근거C", "PKR-IPC 대안 A", "336", r"C14\C14-T2\pkr_ipc_altA.csv", "E-C14T2-03(반영됨)", "", "", "변경 줄. 71.90"),
"U101": (S, "근거C", "PKR-IPC 대안 A", "336", r"C14\C14-T2\pkr_ipc_altA.csv", "E-C14T2-03(반영됨)", "", "", "변경 줄. 73.76"),
"U102": (S, "근거C", "PKR-IPC 대안 A", "336", r"C14\C14-T2\pkr_ipc_altA.csv", "E-C14T2-03(반영됨)", "", "", "변경 줄. 72.82"),
"U103": (S, "근거C", "C 판정", "", r"(MC)\R10_결과_20261005.md 111·113행; C10\C10-T3\summary.txt 5행", "", "", "", "연결 수정 후에도 판정 문구 불변"),
"U104": (S, "데이터", "CA 일치", "280", r"C11\C11-T1\data_check.txt", "E-C11T1-02(반영됨)", "", "", "30,348/30,348·38,380/38,380, N4 98.4%·N7 97.4%"),
"U105": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U106": (S, "근거C", "hyb_pkr 미재계산", "340", r"C10\C10-T3\res\s6_t3_summary.json", "E-C14T2-04(반영됨)", "", "", "변경 줄"),
"U107": (NA, "", "", "", "", "", "", "", "부록 AUC 나열"),
"U108": (NA, "", "", "", "", "", "", "", "규칙 기술"),
"U109": (S, "6트랙", "시드 견고", "319,320", r"C13\C13-T2\seed_bundles.csv, variance_decomp.csv", "E-C13T2-01 대안 행(반영됨)", "", "", ""),
"U110": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U111": (S, "T1", "권리목록 분모", "348", r"C14\C14-T3\cache\s04_out.json", "E-C14T3-05(반영됨)", "", "", "변경 줄. 1,494,298/2,047,575=72.98%"),
"U112": (NA, "", "", "", "", "", "", "", "부록 개수"),
"U113": (S, "트렌드", "선행 시험", "", r"(MC)\_shared_report\fusion_advantage\map5\m3_summary.json", "", "", "", ""),
"U114": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U115": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U116": (NA, "", "", "", "", "", "", "", "부록 목록"),
"U117": (S, "트렌드", "[사후] 표시", "", "", "", "", "", ""),
"U118": (S, "트렌드", "단면 상관", "", r"(MC)\_shared_report\fusion_advantage\split34\t2b_cross_section_prov_vs_cw.csv", "", "", "", ""),
"U119": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U120": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U121": (NA, "", "", "", "", "", "", "", "부록 목록"),
"U122": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U123": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U124": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U125": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U126": (NA, "", "", "", "", "", "", "", "부록 정의"),
"U127": (S, "N8", "확률 과대", "183", r"(MC)\R6_결과_20261004.md", "", "", "", ""),
"U128": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U129": (S, "근거A", "[사후] 표시", "", "", "", "", "", ""),
"U130": (S, "근거A", "순환 판정", "", r"(MC)\R10_결과_20261005.md 136–149행", "", "", "", ""),
"U131": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U132": (S, "T3", "메인그룹 같다", "272,277", r"C11\C11-T2\res\s5_summary_mg__reg.json", "E-C11T2-01(반영됨)", "", "", ""),
"U133": (S, "근거C", "hyb_pkr 미재계산", "340", r"C10\C10-T3\res\s6_t3_summary.json", "E-C14T2-04(반영됨)", "", "", "변경 줄"),
"U134": (NA, "", "", "", "", "", "", "", "부록 숫자"),
"U135": (S, "근거C", "→ 같다", "262", r"C10\C10-T3\summary.txt 31행", "", "", "", ""),
"U136": (S, "근거C", "→ 같다", "262", r"C10\C10-T3\summary.txt 30행", "", "", "", ""),
"U137": (S, "근거C", "R11 판정", "", r"(MC)\_shared_report\label_origin_R10\foreign_public_R11\res_tests.json", "", "", "", ""),
}
assert set(J) == {r["uid"] for r in fr}, set(J) ^ {r["uid"] for r in fr}
out = []
for r in fr:
    v, tr, un, fl, rf, er, fx, pr, nt = J[r["uid"]]
    out.append(dict(uid=r["uid"], line=r["line"], region=r["region"], sec=r["sec"], kind=r["kind"], changed_v15_v16=r["changed_v15_v16"],
                    claim=r["text"], track=tr, unit=un, verdict=v, findings_lines=fl, result_files=rf, errata_ref=er, fix=fx, priority=pr, note=nt))
# 부수 단위(고정 목록 밖, 분모 제외): 79행 괄호 '등록특허 656,360건'
out.append(dict(uid="U003-b", line="79", region="1장", sec="1.4 데이터 연계 프레임워크", kind="부수(분모 밖)", changed_v15_v16="True",
                claim="등록원부도 제공 범위 KR 등록특허 656,360건을 등록번호로 직접 수집", track="데이터", unit="KR 등록 범위", verdict="부수: 표기 오류(경미)",
                findings_lines="352", result_files=r"수집파이프라인\registry_full.sqlite patent(mode=ro): rgst_no '10' 615,042 · '20' 41,318", errata_ref="E-C14T3-D1(원고 밖 D로만 기록)",
                fix="F7", priority=4, note="656,360은 특허 615,042+실용신안 41,318. 부록 503행 DATA-KR-REG도 같음"))
cols = list(out[0].keys())
with open(W + r"\claim_map_ext.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(o) for o in out]
core = [o for o in out if o["uid"] != "U003-b"]
jud = [o for o in core if o["verdict"] != NA]
from collections import Counter
cnt = Counter(o["verdict"] for o in jud)
non = [o for o in jud if o["verdict"] != S]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)
reg = {}
for o in jud:
    reg.setdefault(o["region"], Counter())[o["verdict"]] += 1
st = dict(units=len(core), not_target=sum(1 for o in core if o["verdict"] == NA), judged=len(jud), counts=dict(cnt), nonsupport=len(non),
          nonsupport_rate=round(len(non) / len(jud), 4), wilson95=wilson(len(non), len(jud)), by_region={k: dict(v) for k, v in reg.items()},
          changed_units=sum(1 for o in core if o["changed_v15_v16"] == "True"),
          changed_judged=sum(1 for o in jud if o["changed_v15_v16"] == "True"),
          changed_nonsupport=[o["uid"] for o in non if o["changed_v15_v16"] == "True"], nonsupport_ids=[o["uid"] for o in non])
json.dump(st, open(W + r"\cache\judge_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(st, ensure_ascii=False))

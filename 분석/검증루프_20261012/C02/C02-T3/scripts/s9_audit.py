# C02-T3 1부 숫자 감사: 원고 v3(manuscript_v3.md) 중 C01-T2 미감사 줄의 숫자를 원천 파일·재계산 출력과 대조
# 판정 규칙: PREREG.txt §3–4 (ROUND_HALF_UP, 원고 표기 자릿수)
import sys, json, re, os, sqlite3
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = P + r"\분석"; MC = B + r"\모델경쟁_20260927"; OUT = B + r"\검증루프_20261012\C02\C02-T3"
FA = MC + r"\_shared_report\fusion_advantage"; LO = MC + r"\_shared_report\label_origin_R10"; R12 = LO + r"\confirm_R12"
MS = open(B + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().splitlines()
J = lambda p: json.load(open(p, encoding="utf-8"))
C = lambda p, **k: pd.read_csv(p, encoding="utf-8-sig", **k)
mine = {k: J(OUT + rf"\{k}_out.json") for k in ["s1_map", "s2_rta_sens", "s3_sync", "s4_meta_counts", "s5_misc_meta", "s6_table3", "s7_misc"]}
S1, S3, S4, S5, S6, S7 = (mine[k] for k in ["s1_map", "s3_sync", "s4_meta_counts", "s5_misc_meta", "s6_table3", "s7_misc"])
S2 = mine["s2_rta_sens"]

def line_of(frag, start=1):
    for i in range(start - 1, len(MS)):
        if frag in MS[i]: return i + 1
    raise KeyError(frag)

def ndp(s):
    s = s.replace("−", "-").replace("+", "").replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0

def rnd(x, nd):
    return Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)

def judge(ms, src):
    try:
        m = Decimal(ms.replace("−", "-").replace("+", "").replace(",", ""))
    except Exception:
        return None
    nd = ndp(ms); r = rnd(src, nd); d = abs(r - m) / Decimal(1).scaleb(-nd)
    return "일치" if d == 0 else ("반올림 차" if d == 1 else "불일치")

ROWS = []
def add(sect, frag, ms, src, path, key, verdict=None, memo="", start=1, line=None):
    ln = line or line_of(frag, start)
    v = verdict or (judge(ms, src) if isinstance(src, (int, float, np.floating, np.integer)) else "확인 필요")
    srcs = (str(round(float(src), 6)) if isinstance(src, (float, np.floating)) else str(src))
    ROWS.append(dict(절=sect, 줄=ln, 원고_조각=frag, 원고_값=ms, 원천_값=srcs, 원천_경로=path, 키=key, 판정=v, 메모=memo))
    return ln

# ---------------- 원천 로드 ----------------
conc = J(MC + r"\_shared_report\audit_r7\concordance_resolution_check.json")
t0 = J(MC + r"\T1_산업대기실\TXT_R9\res\t0_txt_agree.json"); t0b = J(MC + r"\T1_산업대기실\TXT_R9\res\t0b_txtfree_agree.json")
c1 = C(FA + r"\c5_R10\c1_label_agreement.csv"); c1w = c1[c1.window.str.startswith("출원2015")].set_index("cc")
fam = J(LO + r"\res_fam_check.json"); rt10 = J(LO + r"\res_tests.json")
r12m = C(R12 + r"\res_main.csv").set_index("기준"); r12s = C(R12 + r"\res_sub.csv"); r12w = C(R12 + r"\res_within.csv").set_index("기준")
r12a = C(R12 + r"\res_app.csv").set_index("기준"); r12p = J(R12 + r"\res_prep.json"); r12t = J(R12 + r"\res_tests.json")
p1 = C(LO + r"\res_p1.csv"); p1d = C(LO + r"\res_p1_disagree.csv"); p3 = C(LO + r"\res_p3.csv"); ph2 = J(LO + r"\res_posthoc2.json")
r11 = C(LO + r"\foreign_public_R11\res_main.csv").set_index("src"); r11t = J(LO + r"\foreign_public_R11\res_tests.json")
c2 = J(FA + r"\c5_R10\c2_results.json"); c5 = J(FA + r"\c5_R10\c5_t3_summary.json")
cs = C(MC + r"\T1_산업대기실\TXT_R9\res\C_summary.csv"); cp = C(MC + r"\T1_산업대기실\TXT_R9\res\C_pool_summary.csv").set_index("pair")
mg = J(MC + r"\T3_기술셀부상예측\TXT_R9\summary_mg_r9.json"); p2r = J(MC + r"\_ideas_R8\p2_results.json")
sf = C(MC + r"\T1_산업대기실\TXT_R9\res\s1_firm_o2019.csv"); sm = J(MC + r"\T1_산업대기실\TXT_R9\res\s1_meta_o2019.json")
nc = C(MC + r"\_shared_report\noncirc_n9n6_R9\results_table.csv")
tv = C(FA + r"\split34\t1_variance_decomposition.csv").set_index("지표"); t2c = C(FA + r"\split34\t2c_panel_two_windows.csv")
t5 = C(FA + r"\split34\t5_cases.csv", dtype={"ksic": str}).set_index("ksic"); t3a = C(FA + r"\split34\t3a_forecast_summary.csv")
t2b = C(FA + r"\split34\t2b_cross_section_prov_vs_cw.csv")
m2 = J(FA + r"\map5\m2_checks.json"); m7 = J(FA + r"\map5\m7_summary.json"); m3 = J(FA + r"\map5\m3_summary.json")
m3c = J(FA + r"\map5\m3c_contemporaneous.json"); m4 = J(FA + r"\map5\m4_summary.json"); m3b = J(FA + r"\map5\m3b_summary.json")
leak = J(MC + r"\T3_기술셀부상예측\VERIFY_R5\leak_t3.json")
els = C(MC + r"\T1_산업대기실\R2\ext_link_summary.csv"); nps = C(MC + r"\_shared\ext\firm_nps_link_rates.csv"); ntl = C(MC + r"\_shared\ext\ntis_link_rates.csv")
ess = C(MC + r"\T3_기술셀부상예측\ESSENTIAL_R7\essential_table_pub.csv", header=None)
n9 = J(MC + r"\N9_정부RnD과제_특허성과예측\final_result_r6.json"); n5 = J(MC + r"\N5_표준특허선언레이더\final_result_r4.json")

# ================= 1.1 =================
S = "1.1"
add(S, "50개 산업이 27개 코드로 뭉쳐", "27", conc["n_concordance_codes_covering_50"], r"(MC)\_shared_report\audit_r7\concordance_resolution_check.json", "n_concordance_codes_covering_50")
add(S, "50개 산업이 27개 코드로 뭉쳐", "34", 50 - conc["ksic_alone_in_code"], "같은 파일", "50 − ksic_alone_in_code(16)")
add(S, "메모리·비메모리·LED·기타 반도체가 모두 C2610", "(문장)", "C2610=" + "/".join(conc["codes_with_2plus"]["C2610"]), "같은 파일", "codes_with_2plus.C2610", "일치", "26111·26112·26121·26129 = 메모리·비메모리·LED·기타 반도체")
add(S, "연계표에 있는 출원의 99.0%는", "99.0", conc["kr_2005_2022"]["share_ipc_multi_industry"] * 100, "같은 파일", "kr_2005_2022.share_ipc_multi_industry", memo="분모: KR 2005–22 출원 중 메인 IPC가 연계표에 있는 87.2%")
add(S, "그룹 수준에서 41.9%", "41.9", t0["group_agree"] * 100, r"(MC)\T1_산업대기실\TXT_R9\res\t0_txt_agree.json", "group_agree", "정의 불일치",
    "숫자는 일치하나 분모가 연계표 배정 문서(95.6%)뿐. 같은 문장의 7.1%·(2.4의)29.4%는 미배정=불일치 전체 기준이고 그 기준 그룹 일치는 40.0%(t0b cw_group_agree_all 0.39997, t0_txt.py 38행). 경미")
add(S, "5자리에서 7.1%만 같다", "7.1", t0["cw5_agree_all(16개만 지목)"] * 100, "같은 파일", "cw5_agree_all")

# ================= 1.2 표 1 =================
S = "1.2 표1"
add(S, "KSIC × IPC 메인그룹, 5,041셀", "5,041", 5041, r"(MC)\T3_기술셀부상예측\VERIFY_R5\robustness_table.md 35행", "R3 공식 5,041셀/504", memo="2순위(실행값 표)")
add(S, "산업 생애주기 9개 사건", "9", 9, r"보고서_20261012\figs\fig02_scorecard_9tracks.csv", "행 수 9")
add(S, "T3: 원점 2021 → 2022–24", "2021→2022–24", "t0=2021 → KR 공개 2022–2024", r"C01\C01-T4\windows_used.csv(T3 final_spec_r3.json)", "", "일치")
add(S, "(AUC 0.794 대 0.642)", "0.794", 0.7935, r"(MC)\T3_기술셀부상예측\VERIFY_R5\robustness_table.md 35행", "AUC 0.7935")
add(S, "(AUC 0.794 대 0.642)", "0.642", 0.6423, "같은 파일", "기준선 0.6423")

# ================= 1.3 =================
S = "1.3"
rc = S4["rows_by_cc"]; tot = S4["rows_total"]
add(S, "모두 15,141,159행이다", "15,141,159", tot, r"(MC)\_shared\meta\*.parquet 직접 집계(scripts\s4_meta_counts.py)", "count(*)")
add(S, "KSIC 세세분류 50개 코드의 특허 목록", "50", S4["n_ksic"], "같은 집계", "count(distinct ksic)")
add(S, "KSIC 라벨은 평균 1.0개", "1.0", S4["kr_labels_per_appl"][0], "같은 집계", "KR 출원당 distinct ksic 평균(최대 1)", memo="전 5개국 (cc,appl_no)당 KSIC 1개, 중복 0(s4 multi_ksic_docs_by_cc=0)")
add(S, "라벨이 87–91% 같다(표 2)", "87–91", "US 0.8731 / JP 0.8696 / EP 0.9047", r"등록번호 패밀리 재연결 독립 구현(s4) = res_fam_check.json links", "끝점 91 ← EP 90.47%", "반올림 차", "EP 90.47%의 정수 사사오입은 90 → '87–90%'(E-C01-08)")
ln = line_of("| KR | 1,087,511 (7.2%)")
for cc, frag in [("KR", "| KR | 1,087,511"), ("US", "| US | 2,382,948"), ("JP", "| JP | 2,163,018"), ("CN", "| CN | 8,700,777"), ("EP", "| EP | 806,905")]:
    l = line_of(frag)
    v = re.search(r"\| ([\d,]+) \(([\d.]+)%\)", MS[l - 1])
    add("1.3 표2", frag, v.group(1), rc[cc], "meta 직접 집계(s4)", f"rows {cc}", line=l)
    add("1.3 표2", frag, v.group(2), rc[cc] / tot * 100, "meta 직접 집계(s4)", f"share {cc}", line=l)
L = line_of("| KR | 1,087,511")
add("1.3 표2", "89.7%(해외 다수결 대비)", "89.7", rt10["P0"]["agree_kr_vs_famMajor"] * 100, r"(MC)\_shared_report\label_origin_R10\res_tests.json", "P0.agree_kr_vs_famMajor", line=L)
add("1.3 표2", "39.2%(해설서 단독)", "39.2", c1w.loc["KR", "free5"] * 100, r"(MC)\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv", "출원2015-19_발명 KR free5", "정의 불일치", line=L,
    memo="값 일치. 단 같은 칸의 55.5%·58.8%는 R12 표본(229,514건)이고 39.2%는 c1 표본(발명 226,098건). R12 표본의 TXT-FREE는 39.31%. 경미")
add("1.3 표2", "55.5%(FREE-IPC)", "55.5", r12p["doc_agree_with_prov"]["H-sg-3"]["eval"] * 100, r"(MC)\_shared_report\label_origin_R10\confirm_R12\res_prep.json", "doc_agree H-sg-3 eval", line=L)
add("1.3 표2", "최대 58.8%", "58.8", max(v["eval"] for k, v in r12p["doc_agree_with_prov"].items() if k != "PKR") * 100, "같은 파일", "후보 33개 eval 최대(C-30-20)", line=L)
for cc, frag in [("US", "| US | 2,382,948"), ("JP", "| JP | 2,163,018"), ("CN", "| CN | 8,700,777"), ("EP", "| EP | 806,905")]:
    l = line_of(frag); cells = [x.strip() for x in MS[l - 1].split("|")]
    agree, repro = cells[3], cells[4]
    if cc != "CN":
        lk = {r[0]: r[4] for r in fam["links"]}
        add("1.3 표2", frag, agree.rstrip("%"), lk[cc] * 100, "등록번호 패밀리 재연결 독립 구현(s4) = res_fam_check.json", f"links {cc}", line=l)
    else:
        add("1.3 표2", frag, "미측정", "CN 연결 6쌍(일치 1/6)", "s4 재연결", "links CN", "일치", "CN은 연결 6쌍뿐이라 '미측정' 표기 타당", line=l)
    add("1.3 표2", frag, repro.rstrip("%"), c1w.loc[cc, "free5"] * 100, r"c1_label_agreement.csv", f"{cc} free5", line=l)
add("1.3", "KR(55.5%)과 해외(48.2–53.5%)의", "48.2–53.5", f"{c1w.loc['JP','free5']*100:.2f}–{c1w.loc['CN','free5']*100:.2f}", "c1_label_agreement.csv", "해외 free5 최소·최대", "일치")
add("1.3", "제공 라벨의 40% 남짓은 공개 제목·IPC로 재현되지 않는다", "40% 남짓", "미재현 KR 44.5%, US 49.0%, JP 51.8%, CN 46.5%, EP 46.9%", "c1 free5, R12 H-sg-3", "1 − 재현율", "불일치",
    "문장 주장 반증(경미): '어느 나라든 40% 남짓'은 JP(51.8%)를 포괄하지 못함 → '45–52%(절반 안팎)'. 결론 영향 없음")

# ================= 1.4 / 표 3 =================
S = "1.4"
add(S, "KR 등록특허 656,360건을 등록번호로", "656,360", S6["registry_tables"]["patent"], r"수집파이프라인\registry_full.sqlite(ro) patent 행 수; meta KR distinct rgst_no도 656,360(s4)", "count")
S = "1.4 표3"
add(S, "50/50 산업¹ (연계표 경유 16/50)", "16", conc["ksic_alone_in_code"], "concordance_resolution_check.json", "ksic_alone_in_code", memo="50/50은 R9 문서(T3NC-LINK, 2순위); 2008 단일 연도 49/50(ext_link_summary)은 각주와 일치")
e = els.set_index("source")
add(S, "KSIC–HS6 1,229쌍 | 48/50 산업", "1,229", int(e.iloc[1]["units"]), r"(MC)\T1_산업대기실\R2\ext_link_summary.csv", "HS6 수입 행 units")
add(S, "KSIC–HS6 1,229쌍 | 48/50 산업", "48", int(e.iloc[1]["ksic_linked"]), "같은 파일", "ksic_linked")
add(S, "50/50 산업 → 36개 IO 소분류", "36", int(e.iloc[0]["units"]), "같은 파일", "IO 행 units")
add(S, "| 385,256행 |", "385,256", S6["corp_applicant_lines"] - 1, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt 직접 행 수(머리행 제외)", "lines−1")
nv = nps.set_index("항목")
add(S, "KR 등록특허의 44.4%(사업자번호 확보분의 93.1%)", "44.4", float(nv.loc["NPS 사업장 매칭 출원인의 특허", "비율"]) * 100, r"(MC)\_shared\ext\firm_nps_link_rates.csv", "NPS 매칭 출원인의 특허 비율")
add(S, "KR 등록특허의 44.4%(사업자번호 확보분의 93.1%)", "93.1", float(nv.iloc[4]["비율"]) * 100, "같은 파일", "사업자번호 확보 특허 대비")
add(S, "산업별 16.0–68.5%", "16.0–68.5", nv.loc["50 KSIC별 NPS 매칭 특허비중 최소/중앙/최대", "값"], "같은 파일", "최소/중앙/최대", "일치")
add(S, "12,322사 / 법인번호 확보 76.2%", "12,322", S6["dart_tables"]["company"], r"공모전데이터\외부데이터\DART\dart_company.sqlite(ro) company 행 수", "count")
add(S, "12,322사 / 법인번호 확보 76.2%", "76.2", "76.2(문서)", r"(MC)\_ideas_R7\IDEAS_R7.md 55행", "2순위 문서", "일치", "원천 표 재계산 안 함(R11 res_tests KIND 2,619행 일치)")
add(S, "과제 1,145,167건", "1,145,167", S6["ntis_project_raw_rows"], r"(MC)\_shared\ext\ntis_project_raw.parquet 행 수", "num_rows")
add(S, "13개 테이블 12,248,048행", "12,248,048", S6["registry_total_rows"], "registry_full.sqlite(ro) 전 테이블 행 합", "sum count", memo="13개 중 unknown_keys는 0행(실데이터 12개 테이블)")
add(S, "13개 테이블 12,248,048행", "13", S6["registry_n_tables"], "같은 DB", "테이블 수")
# 권리이전 재계산
cdb = sqlite3.connect("file:" + (P + r"\수집파이프라인\registry_full.sqlite").replace("\\", "/") + "?mode=ro", uri=True)
ntr = cdb.execute("select count(distinct rgst_no) from owner where ownerRgstCsName is not null and ownerRgstCsName<>''").fetchone()[0]
cdb.close()
add(S, "권리이전 122,974건(18.7%)", "122,974", 122974, "registry_full.sqlite(ro) owner: 소유자 행 2개 이상인 rgst_no 수(=README 정의 재현)", "count(rgst_no having n_owner_rows>1)", "정의 불일치",
    f"숫자는 '소유자 행 2개 이상' 정의로 재현되나 그중 34,651건은 이전 등록사유가 없음(32,309건은 과거 소유자 표시도 없음 = 공동 권리자로 보임). 이전 등록사유(양도·법인분할·합병 등)가 있는 특허는 {ntr:,}건(13.5%), 과거 소유자(finalOwnerYn=N) 포함 시 90,666건(13.8%). '권리이전 발생 18.7%'는 과대")
add(S, "권리이전 122,974건(18.7%)", "18.7", 122974 / 656360 * 100, "README 122,974 / patent 656,360", "비율", "정의 불일치", "이전 등록사유 기준 13.5%(88,323/656,360)")
a = ntl.iloc[0]
add(S, "KR 출원의 9.34%, 등록의 10.95%", "9.34", a["n_linked"] / a["n_patents"] * 100, r"(MC)\_shared\ext\ntis_link_rates.csv", "101,526/1,087,511")
add(S, "KR 출원의 9.34%, 등록의 10.95%", "10.95", a["n_linked_registered"] / a["n_registered"] * 100, "같은 파일", "71,850/656,360")
lk = {r[0]: r[1] for r in S4["fam_links"]}
for cc, v in [("US", "392,048"), ("JP", "216,762"), ("EP", "150,230")]:
    add(S, "US 392,048쌍, JP 216,762쌍", v, lk[cc], "등록번호 패밀리 재연결 독립 구현(s4)", f"links {cc}")
add("1.4 표3 각주", "2008년 단일 연도 파일로는 49/50", "49", int(e.iloc[3]["ksic_linked"]), "ext_link_summary.csv", "KOSIS 2008 ksic_linked")

# ================= 2장 =================
S = "2.1"
add(S, "T3 개발 원점 9개, T1 5개", "9", 9, r"(MC)\T3_기술셀부상예측\R3\FINAL_REPORT_R3.md 24행(2010–2018, 9개)", "2순위")
add(S, "T3 개발 원점 9개, T1 5개", "5", 5, r"(MC)\T1_산업대기실\R3\FINAL_REPORT_R3.md 18행(원점 2008–12)", "2순위")
add(S, "+0.003 이상 오르고 원점의 60% 이상", "+0.003 / 60%", "+0.003 / 60%", r"(MC)\_shared\PROTOCOL_R2.md 26행", "채택 조건", "일치")
add(S, "개발 대비 최종 skill 차이는 −0.005", "−0.005", 0.4229 - 0.4275, "T3 FINAL_REPORT_R3.md(개발 0.4275) / robustness_table(최종 0.4229)", "차")
add(S, "수정 전 값(0.424)", "0.424", 0.424, "T3 FINAL_REPORT_R3.md 25행 0.424 [0.367, 0.473]", "2순위")
S = "2.2"
add(S, "N5 공식 0.165", "0.165", n5["skill_champ_vs_M0"], r"(MC)\N5_표준특허선언레이더\final_result_r4.json", "skill_champ_vs_M0")
add(S, "강한 기준선 대비 0.000", "0.000", 0.0002, r"(MC)\N5_표준특허선언레이더\VERIFY_R5\robustness_table.md 28행", "0.0002 [−0.072, 0.070]")
S = "2.3 표4"
add(S, "T1 최대 차 0.0, T3 누수", "0.0", 0.0, "T1 FINAL_REPORT_R3.md 31행", "교란 최대 차")
add(S, "T1 최대 차 0.0, T3 누수", "0.25–0.48%", "0.25–0.48%", "T3 FINAL_REPORT_R3.md 16행", "", "일치")
add(S, "T3 차이 1e−16, T1 9.0e−8", "1e−16 / 9.0e−8", "1e−16 / 9.0e-8", "T3 FINAL_REPORT_R3.md 118행; T1 FINAL_REPORT_R3.md 19·101행", "", "일치")
add(S, "N8 0/1,000·0/205, N7 0/1,000·0/230, N9 p=0.001", "(숫자 6개)", "N8 0/1,000·0/205; N9 p=0.001; N7 0/1,000·0/230", r"(MC)\R6_결과_20261004.md 56·103·104행", "2순위", "일치")
add(S, "국민연금 키 보정 N7 0.286→0.251", "0.251", 0.251, r"(MC)\N7_특허기업_정부RnD첫수행\VERIFY_R6\robustness_table.md 12행", "0.251 [0.180, 0.324]")
add(S, "N4 0.177→0.140", "0.140", 0.140, r"(MC)\N4_특허행태_기업폐업조기경보\VERIFY_R5\robustness_table.md 84행", "0.140 [0.111, 0.168]")
add("2.3", "강한 기준선 대비 CI가 0에 닿는 T4·N5는", "(문장)", "T4 0.124 [−0.003, 0.208]; N5 0.0002 [−0.072, 0.070]", "T4·N5 VERIFY_R5 robustness_table.md", "", "정의 불일치",
    "T4는 하한이 0에 닿는 수준, N5는 점추정이 0(이득 없음) → E-C01-12. 4.4(416행)와 같은 문제")
S = "2.4"
add(S, "공식 연계표(7.1%)", "7.1", t0["cw5_agree_all(16개만 지목)"] * 100, "t0_txt_agree.json", "cw5_agree_all")
add(S, "(TXT, 29.4%)", "29.4", t0["txt5_agree_all(미배정=불일치)"] * 100, "t0_txt_agree.json", "txt5_agree_all")
add(S, "(TXT-FREE, 40.4%)", "40.4", t0b["free5_agree_all"] * 100, "t0b_txtfree_agree.json", "free5_agree_all")
add(S, "(FREE-IPC, KR 55.5%·해외 48–54%)", "55.5", r12p["doc_agree_with_prov"]["H-sg-3"]["eval"] * 100, "R12 res_prep.json", "H-sg-3 eval")
add(S, "(FREE-IPC, KR 55.5%·해외 48–54%)", "48–54", "48.22–53.48", "c1_label_agreement.csv", "해외 free5", "일치")
add(S, "후보 33개 중 개발 표본 1위인", "33", len(r12p["candidates"]), "R12 res_prep.json", "len(candidates)")
add(S, "후보 33개 중 개발 표본 1위인", "58.8", r12p["doc_agree_with_prov"]["C-30-20"]["eval"] * 100, "R12 res_prep.json", "C-30-20 eval", memo="'공개 최강'은 기준별 개발 1위(NPS5 H-sc-1 56.2%, DART5 H-mg-3 58.1%, KIND3 C-30-20 58.8%); 58.8%는 그중 최대 = 후보 전체 최대이기도 함")
add(S, "개발 AUC가 0.0339 낮아지지만", "0.0339", 0.0339, r"(MC)\_shared_report\fusion_advantage\c5_R10\c5_t3_summary.json(drop_ros 0.7867 − kr_rebuild 0.7529)", "출처표 C-T3-drop_ros-kr_rebuild", memo="AUC 차로는 0.0338; 저장 짝지은 차 0.0339")
add(S, "AUC가 0.788→0.589로", "0.589", float(ess[ess[1] == "conc_feat"].iloc[0][2]), r"(MC)\T3_기술셀부상예측\ESSENTIAL_R7\essential_table_pub.csv", "conc_feat AUC 0.5887")
add(S, "AUC가 0.788→0.589로", "0.788", c5["auc"]["full"], "c5_t3_summary.json auc.full", "0.7877")
S = "2.5"
add(S, "KR 문서의 약 46%가 원점에 미공개", "46", leak["d_appyear_feature_unseen_share"]["2019"]["KR"] * 100, r"(MC)\T3_기술셀부상예측\VERIFY_R5\leak_t3.json", "d_appyear_feature_unseen_share.2019.KR",
    memo="원천 근사식(출원 t0-2 미공개·미공개 영구분 제외). meta 정확 재계산: t0=2019 창 KR 48.0%(발명 48.2%), t0=2020 48.2%(s5). '약 46%'는 1–2%p 과소")
add(S, "34개 산업 성장 순위 예측은", "0.713", float(t3a[(t3a.목표 == "5국") & (t3a.L == 0) & (t3a.구간 == "개발")].쌍순서_M.iloc[0]), r"(MC)\_shared_report\fusion_advantage\split34\t3a_forecast_summary.csv", "5국 L0 개발 쌍순서_M")
add(S, "34개 산업 성장 순위 예측은", "0.523", float(t3a[(t3a.목표 == "5국") & (t3a.L == 2) & (t3a.구간 == "개발")].쌍순서_M.iloc[0]), "같은 파일", "5국 L2 개발(KR L2도 0.523)", memo="L0 개발 원점 2007–14(8), L2 2011–14(4)로 원점 수가 다름")
add(S, "±15% 넘게 어긋난 5개 국가·연도(175/3,255칸)", "5", len(m2["coverage_flagged_total"]), r"(MC)\_shared_report\fusion_advantage\map5\m2_checks.json", "coverage_flagged_total")
add(S, "±15% 넘게 어긋난 5개 국가·연도(175/3,255칸)", "175", m2["coverage_flagged_cells_n"], "같은 파일", "coverage_flagged_cells_n")
add(S, "±15% 넘게 어긋난 5개 국가·연도(175/3,255칸)", "3,255", m2["coverage_cells_n"], "같은 파일", "coverage_cells_n")
add(S, "JP 2021 49,397건→63,886건", "49,397", m7["jp_pub_total_raw_corr_2019_2023"]["2021"][0], r"map5\m7_summary.json", "JP 2021 raw")
add(S, "JP 2021 49,397건→63,886건", "63,886", m7["jp_pub_total_raw_corr_2019_2023"]["2021"][1], "같은 파일", "JP 2021 corr")
add(S, "KSIC 순위 ρ 0.998", "0.998", m7["jp_2021_ksic_rank_raw_vs_corr_spearman"], "같은 파일", "jp_2021_ksic_rank_raw_vs_corr_spearman",
    memo="보정은 공개연도 계열에만 적용(m2_trend_map.py 115행). 문장에 대상 계열 미표기 → E-C01-04 명료화(경고)")

# ================= 3.1.1 =================
S = "3.1.1"
add(S, "국가별·산업별 현시기술우위(RTA, 출원 2017–19년)", "2017–19", "P3=2017–2019", r"map5\m2_trend_map.py 24행", "", "일치")
add(S, "50개 산업 중 25개에서 RTA가 1을 넘고", "25", S1["kr_rta_gt1"], r"map5\m1_counts_appl.parquet에서 동일가중 RTA 재계산(s1_map.py); meta와 행 단위 일치 확인", "KR RTA>1")
add(S, "50개 산업 중 25개에서 RTA가 1을 넘고", "7", S1["leaders"]["KR"], "재계산(s1)", "1위 수 KR")
for cc, v in [("CN", "18"), ("EP", "9"), ("JP", "8"), ("US", "8")]:
    add(S, "(CN 18, EP 9, JP 8, US 8)", v, S1["leaders"][cc], "재계산(s1)", f"1위 수 {cc}")
kl = {k: v for k, _, v in S1["kr_leaders"]}
for k, v, nm in [("20423", "1.72", "화장품"), ("26212", "1.68", "유기 발광"), ("31312", "1.61", "무인 항공기")]:
    add(S, "한국 1위는 화장품(1.72)", v, kl[k], "재계산(s1)", f"KR RTA {k} {nm}")
add(S, "24개 산업은 한국 RTA가 선도국의 0.7배 미만", "24", S1["gap_lt_0.7"], "재계산(s1)", "KR/max < 0.7")
gs = {g[0]: g for g in S1["gap_smallest"]}
add(S, "한의약품(CN의 0.11배)", "0.11", gs["21220"][3], "재계산(s1)", "21220 KR/CN")
add(S, "컴퓨터 프린터(JP의 0.15배)", "0.15", gs["26323"][3], "재계산(s1)", "26323 KR/JP")
add(S, "가장 멀다", "(문장 '가장')", "격차 최소 비율 1·2위 = 21220(0.107), 26323(0.155)", "재계산(s1)", "", "일치")
add(S, "그랜저형 시험 400개 중 다중검정을 통과한 쌍은 0개", "400", 2 * round(m3["expected_false_at_0.05_of_200"] / 0.05), r"map5\m3_summary.json", "방향별 200 × 2", memo="dev_fwd·dev_rev 각 200")
add(S, "그랜저형 시험 400개 중 다중검정을 통과한 쌍은 0개", "0", m3["dev_fwd_q<0.10"] + m3["dev_rev_q<0.10"], "같은 파일", "q<0.10 통과 수")
add(S, "연도 효과를 뺀 상관 US 0.230, JP 0.158", "0.230", S3["n__2004_2018"]["US"][0], r"meta에서 독립 재계산(s3_sync.py) = m3c raw_US 0.2296", "2004–18 US")
add(S, "연도 효과를 뺀 상관 US 0.230, JP 0.158", "0.158", S3["n__2004_2018"]["JP"][0], "같은 재계산 = m3c raw_JP 0.1581", "2004–18 JP")
add(S, "같은 해의 동조만 확인됐다", "(문장 '만')", "풀링 패널 1년 선행 KR→US β0.122(q0.012), JP→KR 0.135(q0.036) 등; 우선권 재출원 제거 후 KR→US 0.058(q0.084), JP→KR 0.108(q0.205)",
    r"map5\m3_summary.json panel_dev, m3b_summary.json panel_dev, SUMMARY.md §4", "", "정의 불일치",
    "산업별 시험(400개)에서는 0개가 맞지만, 산업을 모은 패널에서는 1년 선행이 유의했고 상당 부분이 우선권 재출원 경로. '같은 해의 동조만'은 산업별 기준임을 밝혀야 함(경고)")
add(S, "출원 2017–19년 원계열(보정 없음)", "(캡션)", "fig05 = 원 RTA(최대차 8.9e-16), 보정 0칸", "재계산(s1) vs fig05_rta_map_5x50.csv", "", "일치", "E-C01-01 반영 확인")
add(S, "한국 전체 기술 특화와의 순위상관은 0.45(점추정)로 5개국 중 가장 낮다", "0.45", 0.449, r"C01\C01-T1\summary.txt (c)(i)·verify_1\v1_results.json", "KR 0.449 (US 0.683, JP 0.669, CN 0.757, EP 0.632)", memo="E-C01-03 반영. 직접 재계산 안 함(C01 검증자 확인 항목)")
add(S, "공개 배정으로 바꾸면 한국 1위는 8–16개", "8–16", "cweq_F 8, txt_F 12, free_F 16", r"(MC)\_shared_report\fusion_advantage\c5_R10\c2_results.json landscape_P3", "F 시나리오", "일치",
    "F(5개국 모두 대체) 기준. H(KR 제공·해외 대체)에서는 9–20개(cweq_H 20, txt_H 20, free_H 9) → '5개국 모두 대체 시' 명시 권고(경고)", start=160)
add(S, "한국 RTA 순위상관은 0.20–0.36", "0.20–0.36", "0.199 / 0.247 / 0.360", "c2_results.json rho_vs_prov.KR (F)", "", "일치", start=160)

# ================= 3.1.2 =================
S = "3.1.2"
add(S, "UN Comtrade 한국 수출을 붙였다(48/50개 산업)", "48", m4["export_ksic_n"], r"map5\m4_summary.json", "export_ksic_n")
add(S, "ρ 0.356 [0.026, 0.625]", "0.356", S1["quad_rho_rta_rca"], "재계산(s1: P2 RTA 재생성 × fig06 RCA)", "Spearman n=48")
add(S, "ρ 0.356 [0.026, 0.625]", "0.026", m4["rho_patRTA_expRCA_P2"]["ci"][0], "m4_summary.json", "ci lo")
add(S, "ρ 0.356 [0.026, 0.625]", "0.625", m4["rho_patRTA_expRCA_P2"]["ci"][1], "m4_summary.json", "ci hi")
qc = S1["quad_counts_recomputed"]
for nm, v in [("기술 강·수출 강", "12"), ("기술 강·수출 약", "11"), ("기술 약·수출 강", "3"), ("기술 약·수출 약", "22")]:
    add("3.1.2 표5", f"| {nm} | {v} |", v, qc[nm], "재계산(s1): P2 RTA(m1_counts) × RCA(fig06)", nm)
ex = S1["quad_examples"]
add("3.1.2 표5", "화장품, 합성수지, 발광 다이오드", "(예시)", "모두 기술 강·수출 강 목록에 있음", "재계산(s1)", "", "일치" if all(any(w in n for n in ex["기술 강·수출 강"]) for w in ["화장품", "합성수지", "발광 다이오드"]) else "불일치")
add("3.1.2 표5", "무인 항공기, 감광 재료, 전자 축전기", "(예시)", "모두 기술 강·수출 약", "재계산(s1)", "", "일치" if all(any(w in n for n in ex["기술 강·수출 약"]) for w in ["무인 항공기", "감광 재료", "전자 축전기"]) else "불일치")
add("3.1.2 표5", "산업용 로봇, 전자 부품 실장기판, 영상 기기", "(예시)", "모두 기술 약·수출 강", "재계산(s1)", "", "일치" if all(any(w in n for n in ex["기술 약·수출 강"]) for w in ["산업용 로봇", "실장기판", "영상 기기"]) else "불일치")
add("3.1.2 표5", "한의약품, 컴퓨터 프린터 등", "(예시)", "모두 기술 약·수출 약", "재계산(s1)", "", "일치" if all(any(w in n for n in ex["기술 약·수출 약"]) for w in ["한의약품", "컴퓨터 프린터"]) else "불일치")
add("3.1.2", "판정 불가 2개 제외", "2", m4["quad_tech_export_P2_counts"]["판정불가"], "m4_summary.json", "판정불가")
add("3.1.2", "85.4%가 같은 사분면에 남는다", "85.4", m4["quad_P2_vs_P4_same_share"] * 100, "m4_summary.json", "quad_P2_vs_P4_same_share")
add("3.1.2", "11개 그룹 중 6개", "11", S1["cw_multi_groups_in_fig06"], "재계산(s1)", "다KSIC 연계표 그룹 수")
add("3.1.2", "11개 그룹 중 6개", "6", S1["cw_multi_groups_mixed"], "재계산(s1)", "사분면 섞인 그룹")
add("3.1.2", "관계가 없다(ρ 0.048)", "0.048", m4["rho_patRTAchange_expRCAchange_0817"]["rho"], "m4_summary.json", "rho_patRTAchange_expRCAchange_0817")

# ================= 3.1.3 =================
S = "3.1.3"
add(S, "연계표가 서로 구분하지 못하는 34개 산업", "34", int(tv.loc["g_KR", "n(34중)"]), r"split34\t1_variance_decomposition.csv", "n(34중)")
for k, v, f in [("g_KR", "56.1", "KR 출원 성장 분산의 56.1%"), ("한국기원점유_변화pp", "72.8", "한국 기원 점유 변화의 72.8%"), ("g_5국", "29.1", "5개국 합계 성장은 29.1%뿐"),
                ("x_g_출하액", "59.1", "(출하 59.1%"), ("x_g_사업체수", "47.8", "사업체 47.8%)")]:
    add(S, f, v, tv.loc[k, "그룹안_분산비중_34"] * 100, "t1_variance_decomposition.csv", f"{k} 그룹안_분산비중_34",
        memo=f"무작위 묶음 기대 {tv.loc[k,'무작위묶음_기대']*100:.1f}%, p {tv.loc[k,'p(더_동질)']}")
t2 = t2c[t2c.특허 == "KR출원"].set_index("외부")
for k, v in [("사업체수", "0.41"), ("출하액", "0.35"), ("종사자수", "0.30")]:
    add(S, "사업체 r 0.41, 출하 0.35, 종사자 0.30", v, t2.loc[k, "그룹×창 안 r"], r"split34\t2c_panel_two_windows.csv", f"KR출원×{k}")
add(S, "(n 68)", "68", int(t2.loc["사업체수", "n(KSIC×창)"]), "같은 파일", "n")
cg = "KR출원_연성장%(08-09→16-17)"
for k, col, v, f in [("26211", cg, "−6.6", "액정 표시장치의 KR 출원은 연 −6.6%"), ("26212", cg, "6.2", "유기 발광 표시장치는 +6.2%였고"),
                     ("26211", "KOSIS출하_연성장%", "0.3", "출하도 +0.3% 대 +9.4%"), ("26212", "KOSIS출하_연성장%", "9.4", "출하도 +0.3% 대 +9.4%"),
                     ("26211", "KOSIS사업체_연성장%", "−3.1", "사업체 수도 −3.1% 대 +7.2%"), ("26212", "KOSIS사업체_연성장%", "7.2", "사업체 수도 −3.1% 대 +7.2%"),
                     ("26111", cg, "−14.8", "메모리는 출원 −14.8%"), ("26111", "KOSIS출하_연성장%", "14.2", "출하 +14.2%였다")]:
    memo = ""
    if col == cg:
        cagr = S5["cagr_n"]["26211" if k == "26211" else ("26212" if k == "26212" else "26211")][0]
    add(S, f, v, float(t5.loc[k, col]), r"split34\t5_cases.csv", f"{k} {col}", "정의 불일치" if col == cg and k in ("26211", "26212", "26111") else None,
        "값은 원천과 일치. 원천은 연평균 로그 성장률 [ln(평균16–17+1) − ln(평균08–09+1)]/8 (s2_decomp.py 40행). meta 재계산 복리 연평균(CAGR): 26211 −6.4%, 26212 +6.4%, 26111 약 −13.8%. '연 x%'는 CAGR로 읽힘 → '연평균 로그 성장률' 표기 권고(경고)" if col == cg else
        ("원천은 로그 성장률(s2_decomp.py 66행); 복리 환산 시 출하 26212 +9.9%, 26111 +15.3%" if col == "KOSIS출하_연성장%" else "로그 성장률"))
add(S, "(2008–09년 → 2016–17년)", "2008–09→2016–17", "lg(A,[2008,2009],[2016,2017])", r"split34\s2_decomp.py 58행", "", "일치")
add(S, "34개 전체 단면 상관에서는 연계표 그룹 합계를 넘지 못했고", "(문장)", f"KR 차이 {t2b[t2b.특허=='KR'].차이.min()}~{t2b[t2b.특허=='KR'].차이.max()}, 모든 차이 CI 0 포함", r"split34\t2b_cross_section_prov_vs_cw.csv", "", "일치")
add(S, "같은 코드 C2620 안에서", "(문장 보강)", f"KR 26211+26212 합산 CAGR {S5['cagr_n']['26211+26212'][0]}%, C2620 5개 합산 {S5['cagr_n']['C2620_all'][0]}%", "meta 재계산(s5)", "", "일치",
    "4.1 '거친 코드로 보면 두 추세가 서로 지워진다'를 원천에서 직접 확인(new_result)")

# ================= 3.6 =================
S = "3.6.1"
st = line_of("### 3.6.1")
add(S, "공식 연계표는 34개 산업을 구분하지 못하고", "34", 50 - conc["ksic_alone_in_code"], "concordance json", "50−16", start=st)
add(S, "연계표에 있는 출원의 99.0%가", "99.0", conc["kr_2005_2022"]["share_ipc_multi_industry"] * 100, "concordance json", "share_ipc_multi_industry", start=st)
add(S, "7.1%(연계표), 29.4~40.4%(해설서 텍스트)", "7.1 / 29.4 / 40.4", "7.10 / 29.44 / 40.38", "t0·t0b json", "", "일치", start=st)
add(S, "최대 58.8%(공개 후보 33개 중 1위, 출원 2015–19)", "58.8", r12p["doc_agree_with_prov"]["C-30-20"]["eval"] * 100, "R12 res_prep.json", "C-30-20 eval", start=st)
add(S, "같은 발명의 KR·해외 라벨이 87–91% 같아", "87–91", "90.47(EP)", "s4 재연결", "", "반올림 차", "E-C01-08", start=st)
add(S, "(T3 연계표 대체 0.788→0.589의 대부분)", "0.788→0.589", "0.7877→0.5887", "c5·essential_table_pub", "", "일치", start=st)
S = "3.6.2"; st = line_of("### 3.6.2")
d_ = sf[(sf.기준 == "DART") & sf.조건.str.startswith("(d)") & (sf.가중 == "기업평균")].iloc[0]; n_ = sf[(sf.기준 == "NPS") & sf.조건.str.startswith("(d)") & (sf.가중 == "기업평균")].iloc[0]
add(S, "NPS +10.2%p, DART +25.0%p를 얻었지만(원점 2019)", "10.2", n_["P-FREE"] * 100, r"(MC)\T1_산업대기실\TXT_R9\res\s1_firm_o2019.csv", "NPS (d) 기업평균 P-FREE", start=st)
add(S, "NPS +10.2%p, DART +25.0%p를 얻었지만(원점 2019)", "25.0", d_["P-FREE"] * 100, "같은 파일", "DART (d) 기업평균 P-FREE", start=st)
cand = r12p["candidates"]
ncnt = dict(H=sum(c.startswith("H") for c in cand), D=sum(c.startswith("D") for c in cand), C=sum(c.startswith("C") for c in cand), G=sum(c.startswith("G") for c in cand), T=sum(c.startswith("T") for c in cand))
add(S, "공개 후보 33개(IPC 최빈표 17, 제목+IPC 결합 10, 연계표 그룹 제약 결합 4, 텍스트 단독 2)", "33/17/10/4/2", f"{len(cand)}/{ncnt['H']+ncnt['D']}/{ncnt['C']}/{ncnt['G']}/{ncnt['T']}", "R12 res_prep.json candidates", "", "일치", start=st)
add(S, "재현 점검 2개를 통과했다", "2", "tfidf_argmax_eq_stored 1.0; repro_R11_posthoc2 NPS·DART ok", "R12 res_prep.json, res_tests.json", "", "일치", start=st)
tl = line_of("| NPS 등록업종 | 1,633사", st)
for crit, l in [("NPS5", tl), ("DART5", tl + 1)]:
    cells = [x.strip() for x in MS[l - 1].split("|")]
    r = r12m.loc[crit]
    firms, pats = re.findall(r"[\d,]+", cells[2])[:2]
    add("3.6.2 표9", cells[1], firms, r["기업"], r"R12 res_main.csv", f"{crit} 기업", line=l)
    add("3.6.2 표9", cells[1], pats, r["특허"], r"R12 res_main.csv", f"{crit} 특허", line=l)
    for ci, col in [(3, "PROV"), (4, "BEST값"), (5, "BASE(H-sg-3)"), (6, "T0(TXT-FREE)"), (7, "CWEQ")]:
        add("3.6.2 표9", cells[1], re.search(r"[\d.]+", cells[ci]).group(0), r[col] * 100, "R12 res_main.csv", f"{crit} {col}", line=l)
    nums = re.findall(r"[−+]?[\d.]+", cells[8])
    for vv, col in zip(nums, ["PROV-BEST", "PROV-BEST_lo", "PROV-BEST_hi"]):
        add("3.6.2 표9", cells[1], vv, r[col] * 100, "R12 res_main.csv", f"{crit} {col}", line=l)
    nums = re.findall(r"[−+]?[\d.]+", cells[9])
    for vv, col in zip(nums, ["PROV-T0", "PROV-T0_lo", "PROV-T0_hi"]):
        add("3.6.2 표9", cells[1], vv, r[col] * 100, "R12 res_main.csv", f"{crit} {col}", line=l)
    add("3.6.2 표9", cells[1], "BEST=" + re.search(r"\((H-[a-z]+-\d)\)", cells[4]).group(1), r12t["BEST"][crit], "R12 res_tests.json BEST", "", "일치" if re.search(r"\((H-[a-z]+-\d)\)", cells[4]).group(1) == r12t["BEST"][crit] else "불일치", line=l)
add(S, "DART에서 사후 최강을 골라도 차이는 +6.7%p", "6.7", (r12m.loc["DART5", "PROV"] - r12m.loc["DART5", "사후최강값"]) * 100, "R12 res_main.csv", "PROV − 사후최강값(H-sc-1 0.5617)", start=st)
add(S, "NPS 공개 최강은 평가 표본에서 사후로 고른 최강과 같은 후보", "(문장)", f"BEST {r12m.loc['NPS5','BEST']} = 사후최강 {r12m.loc['NPS5','사후최강(판정X)']}", "R12 res_main.csv", "", "일치", start=st)
L0 = line_of("특허 가중으로 보면 NPS는", st)
for vv, col, crit in [("−3.6", "PROV-BEST_w", "NPS5"), ("−7.6", "PROV-BEST_w_lo", "NPS5"), ("−0.8", "PROV-BEST_w_hi", "NPS5"), ("8.1", "PROV-BEST_w", "DART5"), ("5.9", "PROV-BEST_w_lo", "DART5"), ("10.5", "PROV-BEST_w_hi", "DART5")]:
    add(S, "특허 가중으로 보면 NPS는", vv, r12m.loc[crit, col] * 100, "R12 res_main.csv", f"{crit} {col}", line=L0)
sb = r12s.set_index(["기준", "구간"])
for vv, (crit, g), col in [("81", ("NPS5", ">=300"), "기업"), ("−4.0", ("NPS5", ">=300"), "PROV-BEST"), ("−7.9", ("NPS5", ">=300"), "lo"), ("−0.6", ("NPS5", ">=300"), "hi"),
                           ("58", ("NPS5", ">=20"), "기업"), ("−4.7", ("NPS5", ">=20"), "PROV-BEST"), ("−10.7", ("NPS5", ">=20"), "lo"), ("0.8", ("NPS5", ">=20"), "hi"), ("35", ("DART5", "<10"), "기업")]:
    x = sb.loc[(crit, g)]; x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
    add(S, "특허 가중으로 보면 NPS는", vv, x[col] * (1 if col == "기업" else 100), "R12 res_sub.csv", f"{crit} {g} {col}", line=L0)
dsub = r12s[r12s.기준 == "DART5"]
add(S, "특허 가중으로 보면 NPS는", "+3.5~+10.0", f"{dsub['PROV-BEST'].min()*100:.2f}~{dsub['PROV-BEST'].max()*100:.2f}", "R12 res_sub.csv DART5", "PROV-BEST 범위", "일치", "CI 0 포함은 <10명(35사)·미상(163사)뿐 확인", line=L0)
nsub = r12s[(r12s.기준 == "NPS5") & ~r12s.구간.isin([">=300", ">=20"])]
add(S, "나머지 NPS 구간은 차이가 없다", "(문장)", f"나머지 NPS 구간 CI 모두 0 포함: {bool(((nsub.lo < 0) & (nsub.hi > 0)).all())}", "R12 res_sub.csv", "", "일치" if ((nsub.lo < 0) & (nsub.hi > 0)).all() else "불일치", line=L0)
for vv, crit, col in [("37.2", "NPS5", "PROV"), ("58.9", "NPS5", "BEST"), ("29.4", "NPS5", "RAND"), ("−21.7", "NPS5", "PROV-BEST"), ("−26.4", "NPS5", "PROV-BEST_lo"), ("−16.6", "NPS5", "PROV-BEST_hi"),
                      ("74.5", "DART5", "PROV"), ("69.2", "DART5", "BEST"), ("5.3", "DART5", "PROV-BEST"), ("0.4", "DART5", "PROV-BEST_lo"), ("10.5", "DART5", "PROV-BEST_hi")]:
    add(S, "연계표 그룹이 맞은 문서에서 5자리를 고르는", vv, r12w.loc[crit, col] * 100, "R12 res_within.csv", f"{crit} {col}", line=L0)
L1 = line_of("**3자리 업종(판정 외).**", st)
for vv, crit, col in [("1.8", "NPS3", "PROV-BEST"), ("1.0", "NPS3", "PROV-BEST_lo"), ("2.6", "NPS3", "PROV-BEST_hi"), ("1.2", "DART3", "PROV-BEST"), ("−0.1", "DART3", "PROV-BEST_lo"), ("2.4", "DART3", "PROV-BEST_hi"),
                      ("1.9", "KIND3", "PROV-BEST"), ("−1.4", "KIND3", "PROV-BEST_lo"), ("5.2", "KIND3", "PROV-BEST_hi")]:
    add(S, "**3자리 업종(판정 외).**", vv, r12m.loc[crit, col] * 100, "R12 res_main.csv", f"{crit} {col}", line=L1)
add(S, "**3자리 업종(판정 외).**", "363", r12m.loc["KIND3", "기업"], "R12 res_main.csv", "KIND3 기업", line=L1)
L2 = line_of("**해석 주의(사후 추론).**", st)
add(S, "**해석 주의(사후 추론).**", "56.2", r12p["doc_agree_with_prov"]["H-sc-1"]["eval"] * 100, "R12 res_prep.json", "H-sc-1 eval", line=L2)
L3 = line_of("**참고(판정 외).**", st)
for vv, crit, col in [("−0.7", "NPS5", "PROV_APP-PUB_APP"), ("−3.0", "NPS5", "PROV_APP-PUB_APP_lo"), ("1.5", "NPS5", "PROV_APP-PUB_APP_hi"), ("2.7", "DART5", "PROV_APP-PUB_APP"), ("−0.4", "DART5", "PROV_APP-PUB_APP_lo"), ("5.9", "DART5", "PROV_APP-PUB_APP_hi")]:
    add(S, "**참고(판정 외).**", vv, r12a.loc[crit, col] * 100, "R12 res_app.csv", f"{crit} {col}", line=L3)
add(S, "**참고(판정 외).**", "26.8", sm["NPS"]["share_5digit_in50"] * 100, r"(MC)\T1_산업대기실\TXT_R9\res\s1_meta_o2019.json", "NPS share_5digit_in50", line=L3, memo="분모 NPS 특허 보유 법인 9,054사")
add(S, "**참고(판정 외).**", "15.9", sm["DART"]["share_5digit_in50"] * 100, "같은 파일", "DART share_5digit_in50", line=L3, memo="분모가 DART 기준 8,264사로 NPS(9,054)와 다름 — '특허 보유 법인의 일부'의 분모가 기준마다 다름(경미)")
L4 = line_of("**순환 점검.**", st)
pa = p1[(p1.패밀리 == "가족 다수결") & (p1.가중 == "기업평균")].set_index("기준")
for vv, crit, col in [("26.98", "NPS", "KR라벨"), ("26.83", "NPS", "해외라벨"), ("0.15", "NPS", "KR-해외"), ("−0.87", "NPS", "KR-해외_lo"), ("1.13", "NPS", "KR-해외_hi"),
                      ("61.37", "DART", "KR라벨"), ("62.07", "DART", "해외라벨"), ("−0.69", "DART", "KR-해외"), ("−2.34", "DART", "KR-해외_lo"), ("0.78", "DART", "KR-해외_hi")]:
    add(S, "**순환 점검.**", vv, pa.loc[crit, col] * 100, r"(MC)\_shared_report\label_origin_R10\res_p1.csv", f"{crit} 가족다수결 기업평균 {col}", line=L4)
pp = p3[p3.라벨 == "ksic"].iloc[0]
add(S, "**순환 점검.**", "10.6", pp["b=라벨=출원인최빈"] * 100, "res_p3.csv", "KR 전체 ksic b", line=L4)
add(S, "**순환 점검.**", "65.5", pp["a=라벨=IPC최빈"] * 100, "res_p3.csv", "KR 전체 ksic a", line=L4)
add(S, "**순환 점검.**", "약 1", max(pa.loc["NPS", "KR-해외_hi"], pa.loc["DART", "KR-해외_hi"]) * 100, "res_p1.csv KR-해외 CI 상한 최대(NPS 1.13)", "", "일치", line=L4)
add(S, "**순환 점검.**", "1.84", float(p1d[(p1d.기준 == "DART") & (p1d.패밀리 == "가족 다수결")]["r=KR/해외"].iloc[0]), "res_p1_disagree.csv", "DART 가족다수결 r", line=L4, memo="사전 기준 1.5(RESULT_R10 85행); CI [0.96, 3.14]")
add(S, "**순환 점검.**", "(US 발명자만)", "NPS 30.1 vs 30.4, DART 61.6 vs 62.8, 차이 CI 0 포함", "res_posthoc2.json", "", "일치", line=L4)
L5 = line_of("**정리.** 제공 라벨은 공개 자료로", st)
add(S, "**정리.** 제공 라벨은", "−0.6 [−2.0, +1.0] / 7.3 [+5.1, +9.7]", "−0.62 [−2.04, +1.04] / 7.31 [5.05, 9.65]", "R12 res_main.csv", "", "일치", line=L5)
S = "3.6.3"; st = line_of("### 3.6.3")
mean = cs.groupby("arm").auc.mean(); top = cs.groupby("arm").top1000.mean()
L6 = line_of("AUC는 제공 0.9027", st)
for vv, arm in [("0.9027", "P50"), ("0.9003", "FREE50"), ("0.8994", "TXT50"), ("0.8985", "CW27"), ("0.8973", "NONE")]:
    add(S, "AUC는 제공 0.9027", vv, mean[arm], r"(MC)\T1_산업대기실\TXT_R9\res\C_summary.csv", f"{arm} 5원점 평균", line=L6)
w = [cp.loc[k, "wins"] for k in ["P50-CW27", "P50-TXT50", "P50-FREE50"]]
add(S, "AUC는 제공 0.9027", "5/5", "/".join(w), "C_pool_summary.csv", "P50-CW27·TXT50·FREE50 wins", "일치" if all(x == "5/5" for x in w) else "불일치", line=L6)
for vv, col in [("0.0024", "mean_d"), ("0.0004", "ci_lo"), ("0.0042", "ci_hi")]:
    add(S, "AUC는 제공 0.9027", vv, cp.loc["P50-FREE50", col], "C_pool_summary.csv", f"P50-FREE50 {col}", line=L6)
L7 = line_of("제공 라벨에 공개 배정을 더하면", st)
add(S, "제공 라벨에 공개 배정을 더하면", "0.0007~0.0009", f"{cp.loc['P50TXT-P50','mean_d']:.5f}~{cp.loc['P50FREE-P50','mean_d']:.5f}", "C_pool_summary.csv", "", "일치", line=L7)
add(S, "제공 라벨에 공개 배정을 더하면", "0.0033~0.0040", f"{cp.loc['P50FREE-FREE50','mean_d']:.5f}~{cp.loc['P50TXT-TXT50','mean_d']:.5f}", "C_pool_summary.csv", "", "일치", line=L7)
L8 = line_of("상위 1,000쌍 적중은 18.6% 대 18.2%", st)
add(S, "상위 1,000쌍 적중은 18.6%", "18.6", top["P50"] * 100, "C_summary.csv", "P50 top1000 평균", line=L8)
add(S, "상위 1,000쌍 적중은 18.6%", "18.2", top["FREE50"] * 100, "C_summary.csv", "FREE50 top1000 평균", line=L8)
S = "3.6.4"; st = line_of("### 3.6.4")
L9 = line_of("**재현율.** 해외 4개국", st)
fo = c1w.loc[["US", "JP", "CN", "EP"]]
for vv, col, f in [("3.7–8.1", "cw5", "min-max"), ("29.6–31.1", "txt5", "min-max"), ("48.2–53.5", "free5", "min-max")]:
    lo, hi = vv.split("–")
    add(S, "**재현율.** 해외 4개국", lo, fo[col].min() * 100, "c1_label_agreement.csv", f"해외 {col} 최소", line=L9)
    add(S, "**재현율.** 해외 4개국", hi, fo[col].max() * 100, "c1_label_agreement.csv", f"해외 {col} 최대", line=L9)
add(S, "**재현율.** 해외 4개국", "(JP만 50% 미만)", f"free5 <0.5: {list(fo.index[fo.free5 < 0.5])}", "c1", "", "일치" if list(fo.index[fo.free5 < 0.5]) == ["JP"] else "불일치", line=L9)
L10 = line_of("**예측.** 제공 KSIC 셀로", st)
d_full = c5["auc"]["drop_ros"] - c5["auc"]["kr_rebuild"]
add(S, "**예측.**", "25", (c5["auc"]["hyb_free"] - c5["auc"]["kr_rebuild"]) / 0.0339 * 100, "c5_t3_summary.json", "(hyb_free − kr)/0.0339", line=L10, memo="'25–29%'의 끝점")
add(S, "**예측.**", "29", (c5["auc"]["hyb_txt"] - c5["auc"]["kr_rebuild"]) / 0.0339 * 100, "c5_t3_summary.json", "(hyb_txt − kr)/0.0339", line=L10)
add(S, "**예측.**", "0.0239", c5["pairs"]["drop_ros−hyb_txt"]["dAUC"], "c5_t3_summary.json", "drop_ros−hyb_txt", line=L10)
add(S, "**예측.**", "0.0255", c5["pairs"]["drop_ros−hyb_free"]["dAUC"], "c5_t3_summary.json", "drop_ros−hyb_free", line=L10)
L11 = line_of("**외부 기준.** 우선권 KR", st)
mm = c2["main_abroadRTA_P3_vs_RCA"]
add(S, "**외부 기준.**", "0.545", mm["free"]["rho_prov"], "c2_results.json", "rho_prov", line=L11)
add(S, "**외부 기준.**", "0.444~0.571", f"txt {mm['txt']['rho_alt']} / cweq {mm['cweq']['rho_alt']} / free {mm['free']['rho_alt']}", "c2_results.json", "", "일치", line=L11)
for vv, src_, col in [("4.75", "nps", "PROV-FREE"), ("2.28", "nps", "PROV-FREE_lo"), ("7.24", "nps", "PROV-FREE_hi"), ("11.66", "dart", "PROV-FREE"), ("8.01", "dart", "PROV-FREE_lo"), ("15.44", "dart", "PROV-FREE_hi"),
                      ("−0.33", "kind", "PROV-FREE"), ("−4.96", "kind", "PROV-FREE_lo"), ("4.29", "kind", "PROV-FREE_hi"), ("2.31", "nps", "PROV-PKR"), ("4.04", "dart", "PROV-PKR")]:
    add(S, "**외부 기준.**", vv, r11.loc[src_, col] * 100, r"(MC)\_shared_report\label_origin_R10\foreign_public_R11\res_main.csv", f"{src_} {col}", line=L11)
add(S, "**외부 기준.**", "부분(조건부 유지)", r11t["verdict"]["판정"], "R11 res_tests.json", "", "일치", line=L11)
add(S, "상한 보정 JP 재현율 49.3%", "49.3", 49.3, r"(MC)\R10_결과_20261005.md 50행", "2순위", start=st)
S = "3.6.5 표10"; st = line_of("**표 10.")
rows10 = [("기업 5자리 업종, NPS / DART [사전 등록]", "−0.6 [−2.0, +1.0] / +7.3 [+5.1, +9.7]", "R12 res_main"), ("같은 표본 특허 가중", "−3.6 [−7.6, −0.8] / +8.1 [+5.9, +10.5]", "R12 res_main _w"),
          ("그룹 안 5자리 선택", "−21.7 [−26.4, −16.6] / +5.3 [+0.4, +10.5]", "R12 res_within"), ("T1 새 IPC 서브클래스 첫 출원", "+0.0024 [+0.0004, +0.0042], 5/5", "C_pool P50-FREE50"),
          ("해외 출원 RTA ↔ 수출 RCA", "−0.026 [−0.193, +0.119]", "c2 free d/ci_d"), ("T3, 해외 라벨만 대체", "+0.0239 / +0.0255, 9/9", "c5"), ("해외 라벨 등록업종 일치", "+4.75 / +11.66 / −0.33", "R11 res_main")]
for f, v, s_ in rows10:
    add(S, f, v, "위 본문 대조와 같은 원천값", s_, "", "일치", start=st)
mgp = mg["pairs"]["prov_vs_txt"]
add(S, "T3 메인그룹 성장, AUC", "+0.0020", mgp["dAUC"], r"(MC)\T3_기술셀부상예측\TXT_R9\summary_mg_r9.json", "prov_vs_txt dAUC", start=st)
add(S, "T3 메인그룹 성장, AUC", "−0.0002", mgp["ci"][0], "같은 파일", "ci lo", start=st)
add(S, "T3 메인그룹 성장, AUC", "+0.0041", mgp["ci"][1], "같은 파일", "ci hi", start=st)
add(S, "T3 메인그룹 성장, AUC", "7/9", f"{mgp['wins']}/{mgp['n_orig']}", "같은 파일", "wins", "일치", start=st)
for setn, vv, w_ in [("n9_y_all", "+0.0001", "2/7")]:
    r = nc[(nc.set == setn) & (nc.comparison == "PROV-TXT")].iloc[0]
    add(S, "N9 / N6 분야 무관 특허", vv, r.mean_dAUC, r"(MC)\_shared_report\noncirc_n9n6_R9\results_table.csv", f"{setn} PROV-TXT", start=st)
    add(S, "N9 / N6 분야 무관 특허", w_, f"{r.npos}/{r.n_origins}", "같은 파일", "", "일치", start=st)
r = nc[nc.set.str.startswith("n6") & ~nc.set.str.contains("frozen") & (nc.comparison == "PROV-TXT")]
if len(r):
    r = r.iloc[0]
    add(S, "N9 / N6 분야 무관 특허", "−0.0006", r.mean_dAUC, "results_table.csv", f"{r.set} PROV-TXT", start=st)
    add(S, "N9 / N6 분야 무관 특허", "0/6", f"{r.npos}/{r.n_origins}", "같은 파일", "", "일치", start=st)
cmp_ = p2r["ship"]["compare"]["sig_PROV_KR-sig_TXT"]
add(S, "그룹 안 KOSIS 출하 몫 상승, AUC", "−0.094", cmp_["mean"], r"(MC)\_ideas_R8\p2_results.json", "ship.compare.sig_PROV_KR-sig_TXT", start=st)
add(S, "그룹 안 KOSIS 출하 몫 상승, AUC", "−0.171", cmp_["ci"][0], "같은 파일", "ci lo", start=st)
add(S, "그룹 안 KOSIS 출하 몫 상승, AUC", "−0.012", cmp_["ci"][1], "같은 파일", "ci hi", start=st)
add(S, "그룹 안 KOSIS 출하 몫 상승, AUC", "1/9", cmp_["pos"], "같은 파일", "", "일치", start=st)
add("3.6.5", "외부 기준에서 남는 우위는 DART 업종 일치(+7.3%p)와 T1 라벨 무관 예측(AUC +0.0024)뿐이다", "(문장 '뿐')", "표 10에서 사전 규칙상 '제공 우위'(CI>0) 외부 기준 행 = DART 5자리·DART 가중·DART 그룹 안·T1·해외라벨 NPS/DART(조건부)", "표 10", "", "정의 불일치",
    "같은 표의 해외 라벨 등록업종 NPS +4.75 [+2.28,+7.24]·DART +11.66 [+8.01,+15.44]도 CI>0(판정 '부분(조건부)')이고, DART 특허 가중·그룹 안 선택도 DART 우위. '뿐'은 '사전 등록 주 판정 기준으로' 단서가 필요(경고)")

# ================= 4장·결론·출처 =================
S = "4.1"; st = line_of("## 4.1")
add(S, "(연계표 5자리 7.1%, 해설서 단독 40.4%, 공개 최강 최대 58.8%)", "7.1/40.4/58.8", "7.10/40.38/58.76", "t0·t0b·R12", "", "일치", start=st)
add(S, "DART 기준 7.3%p 높고", "7.3", r12m.loc["DART5", "PROV-BEST"] * 100, "R12 res_main", "DART5 PROV-BEST", start=st)
add(S, "7개에서 8–16개로 바뀐다", "7 / 8–16", "7 / cweq_F 8–free_F 16", "s1·c2_results", "", "일치", "F 시나리오 기준(3.1.1 메모와 같음)", start=st)
add(S, "산업별 선행 시험 400개 중", "400 / 0", "400 / 0", "m3_summary", "", "일치", start=st)
add(S, "KR 출원이 연 −6.6%와 +6.2%로 갈렸다", "−6.6 / +6.2", "t5 로그성장률 −6.6/6.2; CAGR −6.4/+6.4", "t5_cases·meta 재계산", "", "정의 불일치", "3.1.3과 같음(로그 성장률)", start=st)
S = "4.3"; st = line_of("## 4.3")
add(S, "(N8 2.04배 과대)", "2.04", 220.3 / 108, r"(MC)\R6_결과_20261004.md 66행(예측 합 220.3 대 실제 108)", "비", start=st)
add(S, "지금은 9개 트랙 모두 잠글 결과 구간이 소진됐지만", "(문장 '모두')", f"등록원부(right ~{S7['right_max_date']}) 분쟁 예고등록 2025-01~2026-07 {S7['dispute_2025_01_2026_07']}건(2025 93, 2026 41)",
    r"수집파이프라인\registry_full.sqlite(ro) 직접 집계(s7_misc.py); C01-T4 windows_used.csv", "", "정의 불일치",
    "공식 창 길이로는 맞지만, N8(19개월)·T4(8개월, 소멸등록 시차 P97 332일) 단축창과 T3 KR 공개 2025-H1(6개월)은 미채점으로 남아 있음 → E-C01-22 확인")
S = "4.5"
add(S, "이 라벨은 공개 자료로 절반 남짓만 재현되고", "(문장)", "최대 58.8%, FREE-IPC 55.5%", "R12", "", "일치")
add(S, "9개 조기경보가 잠긴 시험을 통과했다", "9", 9, "fig02", "트랙 수")
S = "데이터 출처"
add(S, "특허-산업 융합데이터(KSIC 50개", "15,141,159", tot, "meta 직접 집계(s4)", "count(*)")
add(S, "2026-09-29 수집; 비상업", "2026-09-29", "TOTALHITS 1,094,201 (2026-09-29 기준)", r"(MC)\_shared\ext\ntis_SOURCES.md 13행", "", "일치")
add(S, "분류·보정 | 통계분류포털", "(WIPO 공개건수 보정)", "공개연도 계열 보정에 실제 사용(m2)", "m2_trend_map.py", "", "일치", "E-C01-01 지시대로 유지가 맞음")

# ================= v3에서 새로 들어간 줄(ERRATA 반영분) =================
C1 = B + r"\검증루프_20261012\C01"
ind = J(C1 + r"\C01-T2\verify_0\indep_results.json")["T3"]["p_base"]
add("v3 신규 3.2.2", "동점을 어떻게 자르느냐에 따라 22–36%", "22", ind["top10"]["min"] / 504 * 100, r"C01\C01-T2\verify_0\indep_results.json T3.p_base.top10", "min 113/504")
add("v3 신규 3.2.2", "동점을 어떻게 자르느냐에 따라 22–36%", "36", ind["top10"]["max"] / 504 * 100, "같은 파일", "max 179/504")
add("v3 신규 3.2.2", "상위 100셀에서는 50%가 맞았다(기준선 23–39%)", "23–39", f"{ind['top100']['min']}–{ind['top100']['max']}", "같은 파일", "top100 min–max", "일치", "quicksort 28, stable 33")
lc = C(C1 + r"\C01-T3\label_composition.csv").set_index("key"); rs = C(C1 + r"\C01-T3\rescoring_sensitivity.csv")
add("v3 신규 3.5", "확정 사유로 추정하면 양성의 약 9%(2–29%)다", "9", float(lc.loc["점추정(R1+R2 확률합)", "share_of_108"]) * 100, r"C01\C01-T3\label_composition.csv", "scope_active 점추정 9.63/108")
add("v3 신규 3.5", "확정 사유로 추정하면 양성의 약 9%(2–29%)다", "2–29", f"{float(lc.loc['R1 적극(혼합 포함) 하한','share_of_108'])*100:.1f}–{float(lc.loc['상한(R1 적극+미판정 전부)','share_of_108'])*100:.1f}", "같은 파일", "하한·상한", "일치")
add("v3 신규 3.5", "무효심판만으로 다시 채점해도 상위 1% 포착은 39.3%", "39.3", float(rs[(rs.label == "L_inv") & (rs.metric == "cap_champ")].point.iloc[0]) * 100, r"C01\C01-T3\rescoring_sensitivity.csv", "L_inv cap_champ")
add("v3 신규 3.5", "무효심판만으로 다시 채점해도 상위 1% 포착은 39.3%", "33.3", float(rs[(rs.label == "L_inv") & (rs.metric == "cap_SB")].point.iloc[0]) * 100, "같은 파일", "L_inv cap_SB")
tl = J(C1 + r"\C01-T3\s04_toplist.json")
add("v3 신규 3.5", "포착한 39건 중 30건은 분쟁 이력이 있는 특허의 재분쟁이다", "39 / 30", f"{tl['p_champ']['hits']} / {tl['p_champ']['prior']}", r"C01\C01-T3\s04_toplist.json", "p_champ hits/prior", "일치")
add("v3 신규 4.2 표11", "이력 없는 새 분쟁은 9/78건(과거 분쟁 기준선 4/78)", "9/78, 4/78", f"{tl['p_champ']['new']}/{108-tl['prior_total']}, {tl['p_SB']['new']}/{108-tl['prior_total']}", "같은 파일", "new / (108−prior_total)", "일치")
add("v3 신규 4.2 표11", "모멘텀 기준선 23–39개, 동점 처리에 따라", "23–39", f"{ind['top100']['min']}–{ind['top100']['max']}", r"C01\C01-T2\verify_0\indep_results.json", "top100", "일치")

# ================= 그림 CSV =================
F = B + r"\보고서_20261012\figs"
f2 = C(F + r"\fig02_scorecard_9tracks.csv").set_index("트랙")
chk = {"T3": (0.794, 0.642, 0.423, 0.182), "T1": (0.900, 0.811, 0.473, 0.395), "N8": (0.870, 0.780, 0.409, 0.312), "T4": (0.701, 0.604, 0.246, 0.124),
       "N4": (0.803, 0.760, 0.177, 0.140), "N5": (0.858, 0.830, 0.165, 0.000), "N9": (0.894, 0.826, 0.392, 0.356), "N6": (0.747, 0.703, 0.149, 0.142), "N7": (0.746, 0.645, 0.286, 0.251)}
bad = [t for t, v in chk.items() if any(abs(float(f2.loc[t, c]) - x) > 5e-4 for c, x in zip(["AUC_모형", "AUC_기준선", "skill", "보수적_skill"], v))]
add("그림 3(fig02)", "그림 3. 9개 트랙의 잠긴 최종시험 성적", "(CSV 36값)", f"본문·원천과 다른 칸: {bad or '없음'}", r"figs\fig02_scorecard_9tracks.csv vs 본문(C01 대조값)·N5 robustness(0.8580/0.8300)", "", "일치" if not bad else "불일치",
    "N5 AUC 0.858(0.830)은 본문에 없고 N5 VERIFY_R5 robustness 표와 일치")
f3 = C(F + r"\fig03_necessity_firm_industry_match.csv")
okf3 = abs(f3.iloc[0]["Δ제공_BEST"] - r12m.loc["NPS5", "PROV-BEST"]) < 1e-9 and abs(f3.iloc[1]["Δ제공_BEST"] - r12m.loc["DART5", "PROV-BEST"]) < 1e-9 and abs(f3.iloc[2]["KR라벨"] - pa.loc["NPS", "KR라벨"]) < 1e-9 and abs(f3.iloc[3]["해외라벨"] - pa.loc["DART", "해외라벨"]) < 1e-9
add("그림 8(fig03)", "그림 8. 특허 KSIC 라벨과 기업 실제 업종의 일치", "(CSV)", "res_main·res_p1 값과 동일" if okf3 else "차이 있음", r"figs\fig03 vs R12 res_main.csv, res_p1.csv", "", "일치" if okf3 else "불일치")
f4 = C(F + r"\fig04_within_code_divergence.csv", dtype=str)
f4n = C(F + r"\fig04_within_code_divergence.csv", dtype={"ksic": str}).set_index("ksic")
cols4 = ["KR출원_연성장%(08-09→16-17)", "KOSIS출하_연성장%", "KOSIS사업체_연성장%"]
mx4 = max(float((f4n[c].astype(float) - t5.loc[f4n.index, c].astype(float)).abs().max()) for c in cols4)
add("그림 5(fig04)", "그림 5. 연계표 한 코드 안의 산업 분화", "(CSV)", f"t5_cases.csv와 3개 열 최대차 {mx4} ({len(f4n)}행)", r"figs\fig04_within_code_divergence.csv vs split34\t5_cases.csv", "", "일치" if mx4 < 1e-9 else "불일치", "그림 값도 로그 성장률(본문 3.1.3과 같은 정의 문제)")
f5 = C(F + r"\fig05_rta_map_5x50.csv", dtype={"ksic": str})
add("그림 4(fig05)", "그림 4. 5개국 × 50개 산업", "(CSV 250값)", f"재계산 최대차 {S1['fig05_maxabs_diff']:.1e}, 1위국 일치 {S1['fig05_leader_match']}", r"figs\fig05 vs s1 재계산", "", "일치")
f9 = C(F + r"\fig09_n9_topk_vs_baseline.csv")
v9 = f9.dropna(subset=["모형"]).set_index("모형")
ok9 = abs(v9.loc["파일럿 원판", "상위1%_적중률"] - n9["top1_pilot_orig"]) < 1e-12 and abs(v9.loc["강한 기준선 원판", "상위1%_적중률"] - n9["top1_strong_orig"]) < 1e-12 and abs(v9.loc["챔피언(지연판)", "AUC"] - 0.893920390168637) < 1e-9
add("그림 7(fig09)", "그림 7. N9 정부 R&D 과제의 특허 성과 예측", "(CSV)", "final_result_r6.json 값과 동일" if ok9 else "차이", r"figs\fig09 vs N9 final_result_r6.json", "", "일치" if ok9 else "불일치",
    "그림 (b)는 '강한 기준선 원판 97건·파일럿 원판 96건'을 구분 → 본문 266행 '원판 기준선' 단수 표현(E-C01-10)보다 정확")
f10 = C(F + r"\fig10_validation_design.csv")
add("그림 2(fig10)", "그림 2. 전진 검증과 잠긴 최종시험의 시간 구조", "(CSV)", "T3 개발 0.428→최종 0.423, 9개 원점 2010–18, 최종 t0=2021→2022–24, 미래 2024→2025–27, 위약·감사 숫자 표 4와 동일", r"figs\fig10 vs T3 FINAL_REPORT_R3.md·R6", "", "일치",
    "③ 'N9 동결 해시 6개 일치'는 본문에 없음(그림 전용, R6 §1.1 출처 표기)")
f1 = C(F + r"\fig01_framework.csv")
add("그림 1(fig01)", "그림 1. 제공 데이터를 중심에 둔 세 다리", "(CSV)", "'같은 발명 KR·해외 일치 89.7%', 그룹 41.9%, TXT 29.4%, TXT-FREE 40.4%, 해외 FREE-IPC 48–54%, HS6 48/50", r"figs\fig01_framework.csv", "", "정의 불일치",
    "그림 1 중심 상자의 89.7%는 'KR 문서 기준 해외 다수결 일치'이고 본문(13·60·301행)은 문서쌍 기준 87–91(→87–90)%. 같은 개념에 두 수치가 쓰임 → 그림 문구를 'KR 기준 89.7%(문서쌍 87–90%)'로 통일 권고(경고)")
f7 = C(F + r"\fig07_t3_future_breakout_by_ksic.csv"); f8 = C(F + r"\fig08_t1_waiting_room_index.csv")
add("그림 CSV fig07", "원점 2024년 5,328셀 중 상위 10%인 532셀", "532 / 5,328", f"n_pred_breakout {f7.n_pred_breakout.sum()}, n_cells {f7.n_cells.sum()}", r"figs\fig07_t3_future_breakout_by_ksic.csv 합계", "", "일치" if (f7.n_pred_breakout.sum() == 532 and f7.n_cells.sum() == 5328) else "불일치", "C01 범위(3.2.3)와 겹침 — 합계만 확인")
add("그림 6(fig08)", "대기실 지수'의 합은 9,573.6이다", "9,573.6", float(f8.waiting_room_index.sum()), r"figs\fig08_t1_waiting_room_index.csv 합계", "waiting_room_index 합", memo=f"at_risk 합 {int(f8.at_risk.sum()):,}(본문 1,544,097)")

df = pd.DataFrame(ROWS)
df.to_csv(OUT + r"\audit_numbers_C02_draft.csv", index=False, encoding="utf-8-sig")
print(df.판정.value_counts().to_string())
print(df[df.판정.isin(["확인 필요", "불일치", "반올림 차", "출처 없음"])][["절", "줄", "원고_값", "원천_값", "판정"]].to_string())

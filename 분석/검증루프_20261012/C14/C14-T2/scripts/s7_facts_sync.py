# C14-T2 s7: facts_sync.csv — facts.csv·facts_supplement*.csv 와 v15(부록·본문)·ERRATA 정정 전수 대조. facts는 읽기만.
# 입력: cache/s6_token_diff.csv(자동 토큰 대조), cache/s5_pkr_agree_full.csv(PKR 대안 A), 아래 수동 목록(ERRATA 지정 정정·facts 전용 행·보충 행).
import sys, csv, re
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T2"
def load(fn):
    raw = list(csv.reader(open(R + "\\facts\\" + fn, encoding="utf-8-sig"))); hdr = raw[0]; out = {}
    for i, r in enumerate(raw[1:]):
        if len(r) > len(hdr): r = r[:len(hdr) - 1] + [" / ".join(r[len(hdr) - 1:])]
        r = r + [""] * (len(hdr) - len(r)); d = dict(zip(hdr, r)); d["_file"] = fn; d["_row"] = i + 2; out[d["id"]] = d
    return out
F = {}
for fn in ["facts.csv", "facts_supplement.csv", "facts_supplement_R12.csv"]: F.update(load(fn))
ms = open(R + r"\manuscript_v3.md", encoding="utf-8").read().split("\n")
def appcell(fid):
    for i, l in enumerate(ms):
        if l.startswith("| " + fid + " ") or l.startswith("| " + fid + " ("):
            c = [x.strip() for x in l.strip().strip("|").split("|")]; return i + 1, " | ".join(c[2:-1])
    return "", ""
pk = pd.read_csv(W + r"\cache\s5_pkr_agree_full.csv", encoding="utf-8-sig")
pk = pk[pk.window == "출원2015-19_발명"].set_index("cc")
P = {cc: (round(pk.loc[cc, "pkr5_stored"] * 100, 1), round(pk.loc[cc, "pkr5_reg"] * 100, 1)) for cc in ["US", "JP", "CN", "EP"]}
rows = []
def add(fid, col, cur, v15, errata, status, fix, note=""):
    f = F.get(fid, {}); ln, cell = appcell(fid)
    rows.append(dict(file=f.get("_file", "(없음)"), row=f.get("_row", ""), id=fid, 열=col, facts_현재=cur if cur is not None else f.get("정확한 문장 조각", "")[:300],
                     v15_값=v15 if v15 is not None else (f"부록 {ln}행: {cell[:300]}" if ln else "(v15 부록에 행 없음)"), 근거=errata, 상태=status, 정정안=fix, 비고=note))
# ---- 1) 자동 토큰 대조: facts에만 있는 숫자(옛값) / v15에만 있는 숫자(단서 미동기) ----
T = pd.read_csv(W + r"\cache\s6_token_diff.csv", encoding="utf-8-sig").fillna("")
manual_ids = set()
# ---- 2) ERRATA 지정 정정(수동, 원천 값은 ERRATA 근거 파일에서 확인) ----
M = [
 ("MAP-KR-RTA", "조건·경고", F["MAP-KR-RTA"]["조건·경고"], "본문 168행: 한국 1위 8–16개, 한국 RTA 순위상관 0.19–0.37(동률 규칙에 따라 최대 0.48)", "E-C12T2-11(미반영), E-C13T3-08(반영됨, 본문)", "옛값(조건 열)",
  "조건 열 'ρ 0.20–0.36' → 'ρ 0.19–0.37(동률 규칙에 따라 최대 0.48; C11-T2 대안 A·verify_5)'", "본문은 v15에서 정정됨, facts만 낡음"),
 ("B-P50-CW27", "조건·경고", F["B-P50-CW27"]["조건·경고"], "본문 13·407행: AUC 0.002~0.005", "E-C12T2-11(미반영)", "옛값(조건 열)", "'AUC 0.002~0.004' → 'AUC 0.002~0.005(공개 대안 다섯, C10-T2)'", ""),
 ("B-P50-TXT50", "조건·경고", F["B-P50-TXT50"]["조건·경고"], "본문 13·407행: AUC 0.002~0.005 (부록에서는 693행 '지운 행')", "E-C12T2-11(미반영)", "옛값(조건 열)", "'AUC 0.002~0.004' → 'AUC 0.002~0.005(공개 대안 다섯, C10-T2)'", ""),
 ("B-P50-FREE50", "조건·경고", F["B-P50-FREE50"]["조건·경고"], "본문 13행: 0.002~0.005(해설서 단독 대비는 이원 군집 CI가 0 포함); 334행 [+0.0004, +0.0042] 법인 군집", "E-C12T2-11(미반영), E-C13T3-03(반영됨, 본문)", "옛값(조건 열)·단서 미동기",
  "'AUC 0.002~0.004' → '0.002~0.005(공개 대안 다섯)'; '법인×서브클래스 이원 군집 CI [−0.0006, +0.0057]는 0 포함(C10-T2 boot_summary firm_sc)' 추가", ""),
 ("LINK-IO", "정확한 문장 조각", None, None, "E-C12T2-11(미반영), E-C10T1-04(본문 반영됨)", "옛값", "값 열·문장 조각을 부록 507행 문구('한국은행 2010–2015–2020 접속 산업연관표의 2010년 표(2020 분류, 통합소분류 165; 파일 수정 2026-06-30) × …')로", "facts '산업연관표 2010'은 접속표 이전 표기"),
 ("T3NC-MG", "정확한 문장 조각", None, None, "E-C11T2-05(미반영)", "옛값", "'TXT 0.7947' → 'TXT 0.7950(C11-T2 대안 A)'", ""),
 ("T3NC-prov_vs_txt", "수치·문장 조각·조건", None, None, "E-C11T2-05(미반영), E-C11T2-06(반영됨, 표 10 '같다(경계)')", "옛값",
  "'+0.0020 [−0.0002, +0.0041], 7/9 (같다)' → '+0.0017 [−0.0004, +0.0038], 6/9 (같다(경계))', 조건 열 '연결 정의에 따라 +0.0014~+0.0026, 상속 없음이면 CI 하한>0'", "v15 336행 '(경계)'(E-C13T3-01)와도 맞춤"),
 ("T3NC-txt_vs_none", "정확한 문장 조각", None, "(v15 미사용)", "E-C11T2-05(미반영)", "옛값(facts 전용 행)", "'−0.0023 [−0.0041, −0.0005], 2/9' → '−0.0020 [−0.0039, −0.0003], 2/9'", ""),
 ("T3EXT-prov-txt-사업체수", "정확한 문장 조각", None, "(v15 부록 693행 '지운 행')", "E-C11T2-05(미반영)", "옛값(facts 전용 행)", "'[+0.053, +0.332], 7/9, BH q 0.150' → 'Δ +0.197 [+0.053, +0.333], 7/9, q 0.250'(q는 seed 따라 0.05–0.25)", ""),
 ("T3EXT-prov-txt-종사자수", "정확한 문장 조각", None, "(v15 미사용)", "E-C11T2-05(미반영)", "옛값(facts 전용 행)", "→ 'ρ 0.099 대 −0.0065, +0.106 [−0.029, +0.234], 7/9, q 0.865'", ""),
 ("T3EXT-34", "정확한 문장 조각", None, "(v15 미사용)", "E-C11T2-05(미반영)", "옛값(facts 전용 행)", "→ 'TXT 0.044, +0.111 [+0.0043, +0.214], 7/9'(조건 '경계' 유지)", ""),
 ("C-LAND-free_F", "정확한 문장 조각·조건", None, None, "E-C11T2-05(미반영), E-C10T3-06", "옛값", "'RTA>1 32개 … 0.360' → '31개 … 0.373'; 조건 열 '최빈 동률 규칙 의존(반대 처리 0.473, 균등 분할 0.469)'", "부록 482행은 정정됨"),
 ("C-LAND-txt_F", "정확한 문장 조각", None, None, "E-C11T2-05(미반영)", "옛값", "'RTA>1 28개 … 0.199' → '29개 … 0.195'", "부록 548행은 정정됨"),
 ("C-LAND-free_H", "수치·문장 조각·조건", None, "(v15 부록 행 없음)", "E-C11T2-05(미반영)", "옛값(facts 전용 행)", "수치 9→10, '20/9/4/0.301' → 'RTA>1 21개, 1위 10개, 유지 4, ρ 0.311'; 조건 '동률 반대 0.369, 균등 분할 0.355'", ""),
 ("C-LAND-txt_H", "정확한 문장 조각", None, "(v15 부록 행 없음)", "E-C11T2-05(미반영)", "옛값(facts 전용 행)", "'28/20/5/0.225' → 'RTA>1 27개, 1위 20개, 유지 5, ρ 0.228'", ""),
 ("C-AGREE-US", "수치·문장 조각", None, None, "E-C10T3-01·E-C12T2-07(부록 반영됨), C14-T2 PKR 재계산(신규)", "옛값",
  f"수치 51.0→50.5; 'TXT 30.5%, FREE-IPC 51.0%' → 'TXT 30.0%, FREE-IPC 50.5%'; 진단용 PKR-IPC {P['US'][0]}% → {P['US'][1]}%(대안 A, C14-T2)", "PKR은 부록도 옛값"),
 ("C-AGREE-JP", "수치·문장 조각", None, None, "E-C10T3-01·E-C12T2-07(부록 반영됨), C14-T2", "옛값",
  f"수치 48.2→48.0; 'TXT 29.6%, FREE-IPC 48.2%' → 'TXT 29.4%, FREE-IPC 48.0%'; PKR-IPC {P['JP'][0]}% → {P['JP'][1]}%", "PKR은 부록도 옛값"),
 ("C-AGREE-CN", "문장 조각", None, None, "C14-T2", "동기(변화 없음)", f"PKR-IPC {P['CN'][0]}% = 대안 A {P['CN'][1]}%(2015–19 창 CN 패밀리 연결 0건) — 출처만 보충", ""),
 ("C-AGREE-EP", "문장 조각", None, None, "E-C12T2-07(부록 반영됨), C14-T2", "옛값", f"'TXT 31.1%' → 'TXT 30.7%'; PKR-IPC {P['EP'][0]}% → {P['EP'][1]}%", "PKR은 부록도 옛값"),
 ("C-AGREE-RANGE", "수치·문장 조각", None, None, "E-C09T2-02 부기, E-C10T3(717행), E-C12T2-07(부록 반영됨)", "옛값", "수치 '48–54' → '48–53'; '29.6–31.1%, … 48.2–53.5%' → '29.4–30.7%, … 48.0–53.5%(등록번호 연결 KR 라벨 상속)'", "'48–54'는 이중 반올림"),
 ("C-AGREE-UB", "수치·문장 조각", None, "부록 649행 C-AGREE-PURE: 상속 없는 IPC 표 FREE-IPC US 52.3%, JP 49.4%, EP 55.0%", "E-C10T3-04", "옛값(행 대체)", "C-AGREE-UB(결함 상한 보정 52.3/49.3/55.2, 결과 파일 없음) → C-AGREE-PURE 'US 52.3%, JP 49.4%, EP 55.0%(C10-T3 s2_agree.csv free5_pure)'로 대체", "C-AGREE-PURE는 facts 미수록"),
 ("C-RCA-TXT", "정확한 문장 조각", None, None, "E-C10T3-05(부록 반영됨)", "옛값", "'[0.157, 0.664], +0.100 [−0.120, +0.361]' → '[0.162, 0.652], +0.101 [−0.121, +0.353]'", ""),
 ("C-RCA-FREE-IPC", "수치·문장 조각", None, None, "E-C10T3-05(부록 반영됨)", "옛값", "수치 0.571→0.608; 'ρ 0.571 [0.332, 0.750], −0.026 [−0.193, +0.119]' → 'ρ 0.608 [0.382, 0.772], −0.064 [−0.244, +0.083](공개번호 연결 0.554, 상속 없음 0.596)'", ""),
 ("C-RCA-SUB", "수치·문장 조각", None, "(v15 미사용)", "ERRATA 717행(C10-T3)", "옛값(facts 전용 행)", "보조창 FREE Δ '−0.015' → '−0.024'(C10-T3 대안 A)", "CW-eq +0.091은 duckdb 동점 순서에 따라 +0.087(C10-T3 6-4)"),
 ("C-ABROAD-N", "조건·경고", F["C-ABROAD-N"]["조건·경고"], "(v15 미사용)", "ERRATA 717행, C10-T3 summary 2", "옛값(조건 열)", "'8.9%(5,002건)가 결함 연결로 배정' → '올바른 등록번호 연결 상속 31.1%(공개번호 53.9%); 결함 연결 8.9%는 폐기'", "C10-T3 s2_agree famshare_free_reg 분자 0.3106"),
 ("C-T3-AUCS", "정확한 문장 조각", None, None, "E-C10T3-07(부록 반영됨), C14-T2(신규 경고)", "옛값+결함 잔존", "'hyb_txt 0.7628, hyb_free 0.7613' → '0.7626, 0.7605'; hyb_pkr(진단) 0.7735는 R8 결함 연결 PKR 위 값 → '(R8 연결 기준, 미재계산)' 표시", "부록 519행도 hyb_pkr 옛 경로"),
 ("C-T3-drop_ros-hyb_txt", "수치·문장 조각", None, None, "E-C10T3-07(부록 반영됨)", "옛값", "'+0.0239 [+0.0189, +0.0284]' → '+0.0242 [+0.0194, +0.0289], 9/9'", ""),
 ("C-T3-drop_ros-hyb_free", "수치·문장 조각", None, None, "E-C10T3-07(부록 반영됨)", "옛값", "'+0.0255 [+0.0201, +0.0304]' → '+0.0262 [+0.0212, +0.0308], 9/9'", ""),
 ("C-T3-drop_ros-hyb_pkr", "조건·경고", "", "(v15 미사용)", "C14-T2(신규)", "결함 경로(미재계산)", "조건 열에 'R8 fam_link 상속 PKR 위 값(대안 A 미재계산)' 추가", ""),
 ("C-T3-hyb_txt-kr_rebuild", "수치·문장 조각", None, "(v15 미사용)", "ERRATA 717행", "옛값(facts 전용 행)", "'+0.0100 [+0.0072, +0.0131], 9/9' → '+0.0097 [+0.0069, +0.0127], 9/9'(C10-T3 s6_t3_summary.json reg)", ""),
 ("C-T3-hyb_free-kr_rebuild", "수치·문장 조각", None, "(v15 미사용)", "ERRATA 717행", "옛값(facts 전용 행)", "'+0.0084 [+0.0038, +0.0129], 8/9' → '+0.0077 [+0.0030, +0.0124], 8/9'", ""),
 ("C-T3-RET", "수치·문장 조각", None, None, "E-C10T3-07(부록 반영됨), C14-T2", "옛값+결함 잔존", "수치 '29/25' → '29/23'; '25%(… 71–75% 상실)' → '23%(… 71–77% 상실)'(REG 보존 0.226); hyb_pkr 61%는 R8 결함 경로", ""),
 ("C-RCA-PKR", "조건·경고", "", "(v15 미사용)", "C14-T2(신규)", "결함 경로(미재계산)", "조건 열에 'R8 fam_link 상속 PKR 위 값(대안 A 미재계산; 대안 A에서 분자 상속 비중이 커져 값이 바뀔 수 있음)' 추가", ""),
 ("C-LAND-pkr_H", "조건·경고", "", "(v15 미사용)", "C14-T2(신규)", "결함 경로(미재계산)", "같은 표시", ""),
]
for fid, col, cur, v15, er, st, fix, note in M:
    add(fid, col, cur, v15, er, st, fix, note); manual_ids.add(fid)
# ---- 3) 자동 토큰 대조 나머지 ----
for _, r in T.iterrows():
    if r.id in manual_ids or not r.in_appendix: continue
    if r.only_in_facts and str(r.v15_val).rstrip().endswith("…"):
        add(r.id, "정확한 문장 조각", r.facts_frag, f"부록 {int(r.app_line)}행: {r.v15_val[:300]}", "자동 토큰 대조(C14-T2 s6)", "부록 셀 축약(…)·대조 보류", "v15 부록 셀이 '…'로 잘려 facts 숫자가 빠진 것처럼 보임(옛값 아님)", f"facts에만 {r.only_in_facts}")
    elif r.only_in_facts and r.id == "DATA-REG-EVENTS":
        add(r.id, "정확한 문장 조각", r.facts_frag, f"부록 {int(r.app_line)}행: {r.v15_val[:300]}", "E-C02-T3-01(본문·부록 반영됨), C04-T2 정정 목록", "옛값", "'권리이전 발생 122,974건(18.7%)' → '권리이전 88,323건(13.5%, 이전 등록사유; 소유자 행 2개 이상 122,974건은 공동 권리자 설정등록 약 3.2만 포함)'", "")
    elif r.only_in_facts:
        add(r.id, "정확한 문장 조각", r.facts_frag, f"부록 {int(r.app_line)}행: {r.v15_val[:300]}", "자동 토큰 대조(C14-T2 s6)", "옛값(확인 필요)", f"facts에만 있는 숫자 {r.only_in_facts} → v15 부록 문구로", f"v15에만 {r.only_in_v15}")
    elif r.only_in_v15:
        add(r.id, "정확한 문장 조각", r.facts_frag, f"부록 {int(r.app_line)}행: {r.v15_val[:300]}", "자동 토큰 대조(C14-T2 s6)", "단서 미동기", f"v15 부록에 추가된 단서·숫자 {r.only_in_v15}를 facts 조건 열에", "")
# ---- 4) 보충 행(F 항목) 미추가 ----
S = [("C11-ROS-LINK", "E-C11T3-03"), ("C12-T3-SAB", "E-C12T3-F1"), ("C12-T3-IPI-BREAK", "E-C12T3-F2"), ("C12-N5-SEED", "E-C12T1-F1"),
     ("C13-SEED6", "E-C13T2-F1(부록 534행 반영됨)"), ("C13-N8-CAPTURE", "E-C13T1-F1"), ("C-AGREE-PURE", "부록 649행(E-C10T3-04)"), ("C14-PKR-ALTA", "C14-T2 신규")]
for fid, er in S:
    present = fid in F
    rows.append(dict(file="facts_supplement.csv", row="", id=fid, 열="(행 전체)", facts_현재="(있음)" if present else "(없음)", v15_값=(lambda x: f"부록 {x[0]}행: {x[1][:200]}" if x[0] else "(v15 부록에 행 없음)")(appcell(fid)),
                     근거=er, 상태="동기" if present else "보충 미추가", 정정안="ERRATA 문구 그대로 supplement에 행 추가" if fid != "C14-PKR-ALTA" else
                     f"'C14-PKR-ALTA | 부록 C-AGREE | 진단용 PKR-IPC 대안 A(등록번호 연결 상속): US {P['US'][1]}%, JP {P['JP'][1]}%, CN {P['CN'][1]}%, EP {P['EP'][1]}%(R8 결함 연결 {P['US'][0]}/{P['JP'][0]}/{P['CN'][0]}/{P['EP'][0]}%) | 검증루프_20261012\\C14\\C14-T2\\pkr_ipc_altA.csv'", 비고=""))
# ---- 5) 형식 ----
rows.append(dict(file="facts_supplement_R12.csv", row="7, 12", id="R12-DART5, R12-WITHIN", 열="(형식)", facts_현재="필드 9개(헤더 8개) — 따옴표 없는 쉼표", v15_값="", 근거="C14-T2 s6 파싱", 상태="형식 결함", 정정안="해당 칸을 따옴표로 묶어 8필드로(내용 변경 없음)", 비고="pandas 기본 read_csv가 실패함"))
O = pd.DataFrame(rows)
O.to_csv(W + r"\facts_sync.csv", index=False, encoding="utf-8-sig")
print(O.상태.value_counts().to_string()); print(len(O))
print(O[["file", "row", "id", "상태"]].to_string())

# C18-T3 s3: 숫자 정정·지정 숫자 원천 재검(PREREG 4절). 원천 결과 파일을 직접 열어 다시 계산하고 v19 표기와 비교한다.
# 사용: python -I s3_recheck.py   (sqlite는 mode=ro)
import csv, json, sqlite3, sys, hashlib, re, io
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = AN + r"\검증루프_20261012"; R = AN + r"\보고서_20261012"; W = L + r"\C18\C18-T3"
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
raw = open(R + r"\manuscript_v3.md", "rb").read(); assert hashlib.sha1(raw).hexdigest().startswith("b316df8e")
ms = raw.decode("utf-8").split("\n")
def line_of(s):
    h = [k for k, x in enumerate(ms, 1) if s in x]
    return h
def rnd(x, nd):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
def pct(x, nd=1): return rnd(100 * x, nd)
def rcsv(p):
    return list(csv.DictReader(io.StringIO(open(p, encoding="utf-8-sig").read())))
def rj(p): return json.load(open(p, encoding="utf-8"))
out = []
def chk(cid, item, ms_str, shown, src_vals, src_file, note="", approx=False, kind="지정"):
    """shown: 원고 표기 문자열 리스트, src_vals: 원천을 같은 자릿수로 사사오입한 문자열 리스트"""
    ln = line_of(ms_str)
    present = bool(ln)
    ok = [a == b for a, b in zip(shown, src_vals)]
    if approx and not all(ok):
        ok = [a == b or abs(float(a) - float(b)) <= 10 ** (-(len(a.split(".")[1]) if "." in a else 0)) + 1e-12 for a, b in zip(shown, src_vals)]
    verdict = ("일치" if all(ok) else "불일치") if present else "원고 문자열 없음"
    out.append(dict(ID=cid, 항목=item, v19줄=",".join(map(str, ln)), 원고표기=" / ".join(shown), 원천재계산=" / ".join(src_vals),
                    판정=verdict, 종류=kind, 원천=src_file, 비고=note))
    print(cid, verdict, shown, src_vals)

# 1) 410행 0.464–0.471 (C17-T1 variants.csv)
V = {r["variant"]: r for r in rcsv(L + r"\C17\C17-T1\variants.csv")}
vals = [float(V[k]["skill"]) for k in ["S1N1_re", "S1N1_fx", "CMB_re", "CMB_fx"]]
chk("E-C17T1-01", "410행 원점 정보만 쓰면 skill 범위", "원점 정보만 쓰면 skill 0.464–0.471", ["0.464", "0.471"], [rnd(min(vals), 3), rnd(max(vals), 3)],
    r"C17\C17-T1\variants.csv (S1N1_re·S1N1_fx·CMB_re·CMB_fx)", f"원천 {min(vals):.5f}–{max(vals):.5f}; S1만(C16) 0.4607·0.4671은 범위 밖이나 C17이 원점 정보 전체판으로 대체", kind="반영 숫자")
# 2) 409행 0.38–0.42 (C17-T3 + C05-T1)
T3 = {(r["variant"], r["period"]): r for r in rcsv(L + r"\C17\C17-T3\t3_newipc_sens.csv")}
c17 = [float(T3[(k, "final")]["skill"]) for k in ["V1", "V1r", "V2"]]
C5 = {r["scenario"]: float(r["skill"]) for r in rcsv(L + r"\C05\C05-T1\rescore_scenarios.csv")}
c05 = [C5[k] for k in ["S1", "S1r", "S2", "S2r", "S3"]]
allv = c17 + c05
chk("E-C17T3-01", "409행 신설 분류 셀 빼거나 합쳐도 skill", "빼거나 합쳐도 skill 0.38–0.42", ["0.38", "0.42"], [rnd(min(allv), 2), rnd(max(allv), 2)],
    r"C17\C17-T3\t3_newipc_sens.csv V1·V1r·V2; C05\C05-T1\rescore_scenarios.csv S1–S3", f"C17 {min(c17):.4f}–{max(c17):.4f}, C05(H10만) {min(c05):.4f}–{max(c05):.4f}; 0.38은 H10 외 신설 셀 포함 90셀 재선정판", kind="반영 숫자")
# 3) 242행 39%·약 5%·0.479 (C17-T2)
S2 = rj(L + r"\C17\C17-T2\cache\s2_out.json")
SR = {r["variant"]: r for r in rcsv(L + r"\C17\C17-T2\sens_results.csv")}
off = float(SR["OFF(공식 재현)"]["mean"]); d = float(SR["주(E2, 부트스트랩 p, 1,000시드) 제외"]["mean"])
chk("E-C16T1-04", "242행 이름 매칭 전용 진입 비중·오매칭·제외 skill", "이름 매칭으로만 잡힌 진입(39%)의 약 5%는 오매칭이다(빼면 skill 0.479)",
    ["39", "5", "0.479"], [pct(2990 / 7723, 0), pct(S2["main_E2"]["e"], 0), rnd(off + d, 3)],
    r"C17\C17-T2\cache\s2_out.json main_E2.e; sens_results.csv 주(E2); 2,990/7,723", f"e={S2['main_E2']['e']:.4f} CI {S2['main_E2']['boot_ci'][0]:.4f}–{S2['main_E2']['boot_ci'][2]:.4f}; 0.4727+{d:.4f}={off+d:.4f}", kind="반영 숫자")
# 4) 519행 C-T3-AUCS (C15-T3·C10-T3)
A = rj(L + r"\C15\C15-T3\res\s5_t3_summary.json")["auc_raw"]
chk("E-C15T3-01", "519행 C-T3-AUCS AUC 5개", "hyb_pkr(진단) 0.7755(등록번호 연결·C15-T3)", ["0.7867", "0.7529", "0.7626", "0.7605", "0.7755"],
    [rnd(A["drop_ros"], 4), rnd(A["kr_rebuild"], 4), rnd(A["hyb_txt_reg@C10"], 4), rnd(A["hyb_free_reg@C10"], 4), rnd(A["hyb_pkr_reg"], 4)],
    r"C15\C15-T3\res\s5_t3_summary.json auc_raw", "출처 칸은 c5_t3_summary.json만(E-C16T4-02 미반영)", kind="지정")
# 5) 642행 보존율
den = A["drop_ros"] - A["kr_rebuild"]
ret = [(A[k] - A["kr_rebuild"]) / den for k in ["hyb_txt_reg@C10", "hyb_free_reg@C10", "hyb_pkr_reg"]]
chk("E-C15T3-02", "642행(v18 639행, 과제문 638행) C-T3-RET 보존율·상실", "hyb_pkr 67%(등록번호 연결·C15-T3)", ["29", "23", "71", "77", "67"],
    [pct(ret[0], 0), pct(ret[1], 0), pct(1 - ret[0], 0), pct(1 - ret[1], 0), pct(ret[2], 0)], r"C15\C15-T3\res\s5_t3_summary.json auc_raw로 재계산", f"{ret}", kind="지정")
# 6) 494·516행, 표 2 KR (C16-T3 kr_prefix)
K = rcsv(L + r"\C16\C16-T3\kr_prefix.csv")
def kp(src, prefix, col): return float(next(r for r in K if r["source"].startswith(src) and r["prefix"] == prefix)[col])
n10, n20, nall = kp("meta KR", "10", "n"), kp("meta KR", "20", "n"), kp("meta KR", "ALL", "n")
chk("E-C16T3-02", "494행 C-COUNTRY-SHARE KR 비중·실용신안", "KR 7.2%(1,087,511; 실용신안 50,647 포함)", ["7.2", "1087511", "50647"],
    [pct(nall / 15141159), str(int(nall)), str(int(n20))], r"C16\C16-T3\kr_prefix.csv; 전체 15,141,159행(DATA-META)", kind="지정")
l10 = kp("ntis_patent_flag(LINK", "10", "n_linked"); lall = kp("ntis_patent_flag(LINK", "ALL", "n_linked")
r10 = kp("ntis_patent_flag(LINK", "10", "n_linked_registered"); rall = kp("ntis_patent_flag(LINK", "ALL", "n_linked_registered")
reg10 = kp("ntis_patent_flag(LINK", "10", "n_registered"); regall = kp("ntis_patent_flag(LINK", "ALL", "n_registered")
chk("E-C16T3-01", "516행 LINK-NTIS 연결률(전체·특허만)", "특허만 9.78%), 등록 10.95%(71,850/656,360; 특허만 11.67%)",
    ["1036864", "50647", "9.34", "9.78", "10.95", "11.67", "101526", "71850"],
    [str(int(n10)), str(int(n20)), pct(lall / nall, 2), pct(l10 / n10, 2), pct(rall / regall, 2), pct(r10 / reg10, 2), str(int(lall)), str(int(rall))],
    r"C16\C16-T3\kr_prefix.csv", kind="지정")
chk("표3-③", "96·257행 KR 출원의 9.34%·등록의 10.95%", "KR 출원의 9.34%, 등록의 10.95%", ["9.34", "10.95"], [pct(lall / nall, 2), pct(rall / regall, 2)],
    r"C16\C16-T3\kr_prefix.csv", "분모 실용신안 포함(E-C16T3-S 선택 단서)", kind="본문 대조")
# 7) PKR-IPC (C14-T2)
P = {r["cc"]: r for r in rcsv(L + r"\C14\C14-T2\pkr_ipc_altA.csv") if r["cc"] in ("US", "JP", "CN", "EP")}
chk("E-C14T2-03", "498·499·500·501행 진단용 PKR-IPC(대안 A)", "진단용 PKR-IPC 71.9%(C14-T2 대안 A)", ["71.9", "73.8", "68.0", "72.8"],
    [rnd(float(P[c]["대안A_REG"]), 1) for c in ["US", "JP", "CN", "EP"]], r"C14\C14-T2\pkr_ipc_altA.csv 대안A_REG(독립재현_REG 차 0)", kind="지정")
# 8) 79·503행 656,360 (registry_full.sqlite mode=ro)
con = sqlite3.connect("file:" + REG.replace("\\", "/") + "?mode=ro", uri=True)
cur = con.cursor()
tot = cur.execute("select count(*) from patent").fetchone()[0]
pref = dict(cur.execute("select substr(rgst_no,1,2), count(*) from patent group by 1").fetchall())
con.close()
chk("E-C15T4-07/07b", "79행·503행 제공 범위 KR 등록 656,360(특허·실용신안)", "제공 범위 KR 등록 656,360건(특허 615,042·실용신안 41,318)", ["656360", "615042", "41318"],
    [str(tot), str(pref.get("10")), str(pref.get("20"))], r"수집파이프라인\registry_full.sqlite patent(mode=ro)", f"접두 분포 {pref}", kind="지정")
chk("E-C15T4-07", "79행 '등록특허·실용신안 656,360건'", "KR 등록특허·실용신안 656,360건", ["656360"], [str(tot)], r"registry_full.sqlite(mode=ro)", kind="지정")
# 9) 535행 C16-T1-VIS
V16 = {r["variant"]: r for r in rcsv(L + r"\C16\C16-T1\variants.csv")}
a = V16["S1full_La"]; b = V16["S1fixed_La"]; c = V16["S1full_Lb"]
q1 = None
try:
    q1 = rj(L + r"\C16\C16-T1\cache\q1_pop.json")
except Exception as e:
    q1 = {}
dep = None
for k, v in (q1.items() if isinstance(q1, dict) else []):
    if isinstance(v, (int, float)) and 0.05 < v < 0.1: dep = v
chk("E-C16T1-B1", "535행 C16-T1-VIS", "skill 0.461 [0.442, 0.477](C 법인 행 재계산)·0.467(공식 행 점수 고정)", ["0.461", "0.442", "0.477", "0.467", "0.545"],
    [rnd(a["skill"], 3), rnd(a["ci_lo"], 3), rnd(a["ci_hi"], 3), rnd(b["skill"], 3), rnd(c["skill"], 3)], r"C16\C16-T1\variants.csv",
    f"의존 쌍 비율 7.3%는 q1_pop.json 후보값 {dep}", kind="반영 숫자")
# 10) 536행 C17-T1-NEWC
V1o = rj(L + r"\C17\C17-T1\verify_1\v1_out.json")
n1 = V["N1re"]; r1 = V["R1re"]; r1f = V["R1fx"]; cm = V["CMB_re"]
chk("E-C17T1-B1", "536행 C17-T1-NEWC", "N1 Δ +0.0055 [+0.0015, +0.0098](무작위 동수 제외 +0.0052로 같음)",
    ["232", "1304", "0.0055", "0.0015", "0.0098", "0.0052", "0.0056", "0.0001", "0.471", "0.452", "0.491"],
    [str(V1o["N1_vis"]), str(V1o["newc_firms"]), rnd(n1["delta_skill"], 4), rnd(n1["dci_lo"], 4), rnd(n1["dci_hi"], 4), rnd(V1o["rand_drop_same_nfirms_delta"]["mean"], 4),
     rnd(r1["delta_skill"], 4), rnd(r1f["delta_skill"], 4), rnd(cm["skill"], 3), rnd(cm["ci_lo"], 3), rnd(cm["ci_hi"], 3)],
    r"C17\C17-T1\variants.csv; verify_1\v1_out.json", kind="반영 숫자")
# 11) 537행 C17-T2-ALIAS
ci = S2["main_E2"]["boot_ci"]; sr = SR["주(E2, 부트스트랩 p, 1,000시드) 제외"]
vv = [float(SR[k]["mean"]) for k in SR if k.startswith("변형 ") or k.startswith("반박 R1") or k.startswith("\"V9") or k.startswith("V9")]
chk("E-C17T2-B1", "537행 C17-T2-ALIAS", "오매칭 5.0% [4.0, 6.3](코드 대조 E2 보정; 코드 확정 20쌍 0.7%)",
    ["5.0", "4.0", "6.3", "0.7", "0.0066", "0.0041", "0.0087", "0.0002", "0.0036", "0.0087"],
    [pct(S2["main_E2"]["e"]), pct(ci[0]), pct(ci[2]), pct(S2["lower_code_only"]["rate"]), rnd(sr["mean"], 4), rnd(sr["p2_5"], 4), rnd(sr["p97_5"], 4),
     rnd(SR["하한판: 코드 확정 오매칭만 제외"]["mean"], 4), rnd(min(vv), 4), rnd(max(vv), 4)], r"C17\C17-T2\cache\s2_out.json; sens_results.csv",
    "변형 범위는 변형·반박 R1·V9 포함", kind="반영 숫자")
# 12) 538행 C17-T3-NEWIPC
f0 = T3[("V0", "final")]; f1 = T3[("V1", "final")]
ncell = int(float(f0["n"]) - float(f1["n"])); npos = int(float(f0["pos"]) - float(f1["pos"]))
chk("E-C17T3-B1", "538행 C17-T3-NEWIPC", "원점 이후 IPC 메인그룹 셀 90(1.8%, 양성 34)", ["90", "1.8", "34", "0.406", "0.380", "0.400", "0.423"],
    [str(ncell), pct(ncell / 5041), str(npos), rnd(f1["skill"], 3), rnd(T3[("V1r", "final")]["skill"], 3), rnd(T3[("V2", "final")]["skill"], 3), rnd(f0["skill"], 3)],
    r"C17\C17-T3\t3_newipc_sens.csv", kind="반영 숫자")
chk("표6", "217행 신설 IPC 셀 제외 0.406", "| 신설 IPC 셀 제외 | 0.406 |", ["0.406"], [rnd(f1["skill"], 3)], r"C17\C17-T3\t3_newipc_sens.csv V1 = C05 0.40640", kind="본문 대조")
# 13) 388행 표 11 기술이전 (C15-T1 s4_horizon, C02-T1)
H = rcsv(L + r"\C15\C15-T1\cache\s4_horizon.csv")
def hz(h, g, col="regdate"): return float(next(r for r in H if r["h"] == str(h) and r["group"] == g)[col])
lap = rj(L + r"\C02\C02-T1\_tab_e_lapse.json")
st = lap["standardized"][0]
chk("E-C16T4-01", "388행 5년 내 포기 14.2%(비기원 21.0%), 격차 1.4–3.5%p, 10년 내 역전", "5년 내 포기 14.2%(비기원 21.0%; 유형 표준화 격차 1.4–3.5%p, 10년 내 역전)",
    ["14.2", "21.0", "1.4", "3.5", "역전"], [pct(hz(5, "정부R&D 기원")), pct(hz(5, "비기원")), rnd(st["gap_pp_govmix"], 1), rnd(st["gap_pp_nongovmix"], 1),
     "역전" if hz(10, "정부R&D 기원") > hz(10, "비기원") else "역전 아님"], r"C15\C15-T1\cache\s4_horizon.csv(regdate); C02\C02-T1\_tab_e_lapse.json standardized[atype]",
    f"10년 regdate 기원 {hz(10,'정부R&D 기원'):.4f} 대 비기원 {hz(10,'비기원'):.4f}", kind="반영 숫자")
chk("E-C15T1-02", "655행 NTIS-T3-SUM 7·10년", "7년 내 37.7% 대 35.2%, 10년 내(등록 2007–15) 55.5% 대 52.2%", ["37.7", "35.2", "55.5", "52.2"],
    [pct(hz(7, "정부R&D 기원")), pct(hz(7, "비기원")), pct(hz(10, "정부R&D 기원")), pct(hz(10, "비기원"))], r"C15\C15-T1\cache\s4_horizon.csv(regdate)", kind="반영 숫자")
chk("NTIS-T3-SUM", "655행 이전 10.5% 대 11.5%", "이전 10.5% 대 11.5%", ["10.5", "11.5"], [pct(lap["recompute"]["transfer5_gov"]), pct(lap["recompute"]["transfer5_nongov"])],
    r"C02\C02-T1\_tab_e_lapse.json recompute", kind="본문 대조")
# 14) 143행 0.0022·0.007, 0.25–0.26·0.20–0.21
S10 = {r["variant"]: r for r in rcsv(L + r"\C10\C10-T1\sens_results.csv")}
S14 = rcsv(L + r"\C14\C14-T3\sens_results.csv")
UP14 = float(next(r for r in S14 if r["variant"] == "UPPER")["final_d_skill"])   # 사전 등록 판정 지표(max(|S3 평균|,|S4|,|AN_B−A|))
mx14 = -UP14
seedmin14 = min(float(r["final_d_skill"]) for r in S14 if r["variant"].startswith("S3_") and r["variant"] != "S3_mean")
OV = rcsv(L + r"\C02\C02-T2\official_vs_provided.csv")
kr = [float(r["ratio_pat"]) for r in OV if r["country"] == "KR" and 2012 <= int(r["year"]) <= 2023]
cn = [float(r["ratio_inv_rgst"] or "nan") for r in OV if r["country"] == "CN" and 2018 <= int(r["year"]) <= 2022]
chk("143행", "143행 T1 물가·IO 각각 빼도 하락 0.0022 이하, 권리목록 영향 0.007 이하", "각각 빼도 T1·T3 skill 하락 0.0022 이하", ["0.0022", "0.007"],
    [rnd(abs(min(float(S10["S4"]["d_skill"]), float(S10["S5"]["d_skill"]))), 4), rnd(abs(mx14), 3)], r"C10\C10-T1\sens_results.csv S4·S5; C14\C14-T3\sens_results.csv UPPER(판정 지표)",
    f"S4 {float(S10['S4']['d_skill']):.5f}, S5 {float(S10['S5']['d_skill']):.5f}, C14 판정 지표 {UP14:.5f}(S3 시드 평균); 개별 섭동 시드 최대 하락 {seedmin14:.5f} → '0.007 이하'는 판정 지표 기준(warning: 시드 단위로는 0.008)", kind="본문 대조")
chk("143행b", "143행 공식 대비 비율 KR 0.25–0.26(2012–23)", "KR 등록 2012–23년 0.25–0.26", ["0.25", "0.26"], [rnd(min(kr), 2), rnd(max(kr), 2)],
    r"C02\C02-T2\official_vs_provided.csv ratio_pat", f"{min(kr):.4f}–{max(kr):.4f}", kind="본문 대조")
# 15) 482·168행 (C11-T2 verify_5)
v5 = rj(L + r"\C11\C11-T2\verify_5\v5_out.json"); vf = rj(L + r"\C11\C11-T2\verify_5\v5_frac.json")
rg = v5["range_168"]
chk("168·482행", "168행 0.19–0.37·최대 0.48, 482행 0.373·0.473·0.476", "표 동률만 반대 0.473, 배정까지 반대 0.476", ["0.19", "0.37", "0.48", "0.373", "0.473", "0.476", "31"],
    [rnd(rg["REG_base"]["min_raw"], 2), rnd(rg["REG_base"]["max_raw"], 2), rnd(max(x["max_raw"] for x in rg.values()), 2), rnd(vf["asc_free_F"]["rho"], 3),
     rnd(vf["desc_free_F"]["rho"], 3), rnd(rg["REG_tie_desc_full"]["max_raw"], 3), str(vf["asc_free_F"]["kr_rta_gt1"])], r"C11\C11-T2\verify_5\v5_out.json, v5_frac.json", kind="반영 숫자")
# 16) 485·488·581·534행 (C15-T2 seed_bundles)
SB = rcsv(L + r"\C15\C15-T2\seed_bundles.csv")
def sb(tr, col): return [float(r[col]) for r in SB if r["track"] == tr and r[col] not in ("", None)]
t1s, t1lo, t3s = sb("T1", "skill"), sb("T1", "ci_lo"), sb("T3", "skill")
t1top = [float(r["topk2"]) for r in SB if r["track"] == "T1"]
t1sl = sb("T1", "strong_ci_lo"); t3sl = sb("T3", "strong_ci_lo")
chk("E-C15T2-02/01", "485·488·581·534행 시드 10묶음 범위", "다른 학습 시드 10묶음: 0.472–0.475, 하한 0.453–0.456",
    ["0.472", "0.475", "0.453", "0.456", "0.423", "0.432", "29.8", "31.6", "0.376", "0.384", "0.124", "0.143"],
    [rnd(min(t1s), 3), rnd(max(t1s), 3), rnd(min(t1lo), 3), rnd(max(t1lo), 3), rnd(min(t3s), 3), rnd(max(t3s), 3), rnd(min(t1top) / 10, 1), rnd(max(t1top) / 10, 1),
     rnd(min(t1sl), 3), rnd(max(t1sl), 3), rnd(min(t3sl), 3), rnd(max(t3sl), 3)], r"C15\C15-T2\seed_bundles.csv", f"T1 n={len(t1s)}, T3 n={len(t3s)}", kind="반영 숫자")
# 17) 미반영 숫자 변경 제안: E-C09T2-04(142행 '약 46–48%'→'약 47–48%')
U = rj(L + r"\C03\C03-T3\_s8_unseen.json")
uv = [U[k]["unseen_min_pub_rgst"] for k in U]
chk("E-C09T2-04", "142행 원점 미공개 비율(제안 '약 47–48%')", "약 46–48%가 원점에 미공개", ["47", "48"], [pct(min(uv), 0), pct(max(uv), 0)],
    r"C03\C03-T3\_s8_unseen.json", f"원천 {min(uv)*100:.2f}–{max(uv)*100:.2f}%: 제안값 47–48 일치, 현행 '약 46–48%'도 '약' 허용 범위(PREREG 4)", kind="미반영 제안")
# 18) E-C07T1-08(518행 49/50 → 50/50, 5,925/5,925)
OC = rcsv(L + r"\C07\C07-T1\official_compare.csv")
n_ok = sum(1 for r in OC if r["status"] == "일치(≤0.5%)")
n_all = sum(1 for r in OC if r["status"] in ("일치(≤0.5%)",) or r["status"].startswith("불일치"))   # 비교 가능 칸(합계·비교불가·한쪽만·공식칸없음 제외)
codes = {r["code"] for r in OC if r["code"] not in ("0", "C")}
chk("E-C07T1-08", "518행 LINK-KOSIS5(제안: 50/50, 공식값 대조 5,925/5,925)", "KOSIS 광업제조업조사 5자리(2008): KSIC5 동일코드 49/50", ["5925", "5925", "50"],
    [str(n_ok), str(n_all), str(len(codes))], r"C07\C07-T1\official_compare.csv", f"현행 49/50(2008년 값)도 사실(본문 각주와 같음) — 제안은 범위 확장; 전체 {len(OC)}행 중 비교 가능 {n_all}칸(합계·비교불가·한쪽만·공식칸없음 제외)", kind="미반영 제안")
# 19) 선택 참고판 덧붙임 숫자: E-C15T1-03(515행), E-C17T3-B2(DATA-NTIS-PJ 출처)
s2o = rj(L + r"\C15\C15-T1\cache\s2_out.json"); ev = s2o["event_counts_by_prefix"]
tot_ext = sum(ev[k]["10"] + ev[k]["20"] for k in ("NP", "EXP", "EXT_UNK"))
chk("E-C15T1-03", "515행 소멸등록 349,507(실용신안 포함; 특허 불납 293,169·만료 20,112)", "소멸등록 349,507", ["349507", "293169", "20112"],
    [str(tot_ext), str(ev["NP"]["10"]), str(ev["EXP"]["10"])], r"C15\C15-T1\cache\s2_out.json event_counts_by_prefix(NP+EXP+EXT_UNK)", kind="미반영 제안")
NC = rcsv(L + r"\C17\C17-T3\cache\ntis_count_check.csv")
rat = [float(r["ratio_n"]) for r in NC if r["ratio_n"]]
chk("E-C17T3-B2", "DATA-NTIS-PJ 출처 칸 덧붙임(원문 대조 0.99993–1.00000)", "NTIS 과제 2002–2024 1,145,167건", ["0.99993", "1.00000"], [rnd(min(rat), 5), rnd(max(rat), 5)],
    r"C17\C17-T3\cache\ntis_count_check.csv", "2018–24년", kind="미반영 제안")
# 20) 130·409행 T4 강화 하한 −0.003 (E-C13T2-02 '0 밑')
chk("E-C13T2-02", "130·409행 T4 강화 기준선 대비 CI 하한(제안 '0 밑')", "13변수 로지스틱 강화 기준선(0.659) 대비 skill 0.124 [−0.003, 0.208]", ["−0.003"], ["−0.003"],
    r"부록 533행 OFF-T4-STRONG; C13\C13-T2 seed_bundles(하한 −0.016~−0.003 모두 0 아래)", "제안 '0 밑'은 사실과 맞음(현행 '0에 닿고'도 오류 아님)", kind="미반영 제안")

with open(W + r"\recheck_numbers.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
from collections import Counter
print(Counter(o["판정"] for o in out))
json.dump(out, open(W + r"\cache\s3_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

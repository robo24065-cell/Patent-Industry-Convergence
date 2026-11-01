# C18-T3 verify 2: 숫자 정정 전수 독립 재판정(PREREG 8절). s3_recheck.py·캐시를 쓰지 않고 pandas(csv)·json·sqlite(다른 질의)로 원천을 다시 읽는다.
# 원고 표기는 v19(manuscript_v3.md, b316df8e)에서 정규식으로 직접 뽑는다(s3처럼 손으로 적은 표기 목록을 쓰지 않음).
# 추가: s3가 재계산하지 않은 3개(143행 CN 0.20–0.21, 130·409·533행 T4 강화 하한 −0.003, 본문 토큰 스캔 미대응 중 원천 경로가 확인되는 7개).
import re, sys, json, sqlite3, hashlib
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = AN + r"\검증루프_20261012"; R = AN + r"\보고서_20261012"; W = L + r"\C18\C18-T3"; MC = AN + r"\모델경쟁_20260927"
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
raw = open(R + r"\manuscript_v3.md", "rb").read(); assert hashlib.sha1(raw).hexdigest().startswith("b316df8e")
MS = raw.decode("utf-8").split("\n")
def r_(x, nd):  # 사사오입
    return str(Decimal(str(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def grab(line, pat, k=1):
    m = re.search(pat, MS[line - 1]); assert m, (line, pat)
    return [m.group(i) for i in range(1, k + 1)]
def jl(p): return json.load(open(p, encoding="utf-8"))
def csvp(p): return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
OUT = []
def rec(cid, line, shown, src, path, note="", approx=False):
    shown = [s.replace(",", "").replace("−", "-") for s in shown]; src = [s.replace("−", "-") for s in src]
    ok = all(a == b for a, b in zip(shown, src)) and len(shown) == len(src)
    if not ok and approx and len(shown) == len(src):   # PREREG 4: '약' 표기는 끝자리 ±1 허용
        ok = all(abs(float(a) - float(b)) <= 10 ** -(len(a.split(".")[1]) if "." in a else 0) + 1e-12 for a, b in zip(shown, src))
    OUT.append(dict(ID=cid, 줄=line, 원고=" / ".join(shown), 원천=" / ".join(src), 독립판정="일치" if ok else "불일치", 원천경로=path, 비고=note))
    print(cid, line, "OK" if ok else "MISMATCH", shown, src)
# 1 E-C17T1-01 410
v = csvp(L + r"\C17\C17-T1\variants.csv").set_index("variant")
vals = [float(v.loc[k, "skill"]) for k in ["S1N1_re", "S1N1_fx", "CMB_re", "CMB_fx"]]
rec("E-C17T1-01", 410, grab(410, r"원점 정보만 쓰면 skill (0\.\d+)–(0\.\d+)", 2), [r_(min(vals), 3), r_(max(vals), 3)], r"C17-T1\variants.csv")
# 2 E-C17T3-01 409
t3 = csvp(L + r"\C17\C17-T3\t3_newipc_sens.csv"); t3f = t3[t3["period"] == "final"].set_index("variant")
c5 = csvp(L + r"\C05\C05-T1\rescore_scenarios.csv").set_index("scenario")
allv = [float(t3f.loc[k, "skill"]) for k in ["V1", "V1r", "V2"]] + [float(c5.loc[k, "skill"]) for k in ["S1", "S1r", "S2", "S2r", "S3"]]
rec("E-C17T3-01", 409, grab(409, r"합쳐도 skill (0\.\d+)–(0\.\d+)", 2), [r_(min(allv), 2), r_(max(allv), 2)], r"C17-T3\t3_newipc_sens.csv + C05-T1\rescore_scenarios.csv",
    f"min {min(allv):.4f} max {max(allv):.4f}")
# 3 E-C16T1-04 242
s2 = jl(L + r"\C17\C17-T2\cache\s2_out.json"); sr = csvp(L + r"\C17\C17-T2\sens_results.csv").set_index("variant")
off = float(sr.loc["OFF(공식 재현)", "mean"]); dd = float(sr.loc["주(E2, 부트스트랩 p, 1,000시드) 제외", "mean"])
g = grab(242, r"진입\((\d+)%\)의 약 (\d+)%는 오매칭이다\(빼면 skill (0\.\d+)\)", 3)
rec("E-C16T1-04", 242, g, [r_(100 * 2990 / 7723, 0), r_(100 * s2["main_E2"]["e"], 0), r_(off + dd, 3)], r"C17-T2\cache\s2_out.json, sens_results.csv", f"e={s2['main_E2']['e']:.4f}")
# 4·5 519·642
A = jl(L + r"\C15\C15-T3\res\s5_t3_summary.json")["auc_raw"]
ln519 = next(i for i, s in enumerate(MS, 1) if s.startswith("| C-T3-AUCS |"))
g = grab(ln519, r"제공 (0\.\d+), KR만 (0\.\d+), hyb_txt (0\.\d+), hyb_free (0\.\d+), hyb_pkr\(진단\) (0\.\d+)", 5)
rec("E-C15T3-01", ln519, g, [r_(A[k], 4) for k in ["drop_ros", "kr_rebuild", "hyb_txt_reg@C10", "hyb_free_reg@C10", "hyb_pkr_reg"]], r"C15-T3\res\s5_t3_summary.json")
ln642 = next(i for i, s in enumerate(MS, 1) if s.startswith("| C-T3-RET |"))
g = grab(ln642, r"hyb_txt (\d+)%, hyb_free (\d+)%\(공개 라벨로 바꾸면 해외 신호의 (\d+)–(\d+)% 상실\), hyb_pkr (\d+)%", 5)
b, t = A["kr_rebuild"], A["drop_ros"]
ret = lambda k: (A[k] - b) / (t - b)
rec("E-C15T3-02", ln642, g, [r_(100 * ret("hyb_txt_reg@C10"), 0), r_(100 * ret("hyb_free_reg@C10"), 0), r_(100 * (1 - ret("hyb_txt_reg@C10")), 0), r_(100 * (1 - ret("hyb_free_reg@C10")), 0), r_(100 * ret("hyb_pkr_reg"), 0)], r"C15-T3 auc_raw 재계산")
# 6 494·516·96 (kr_prefix)
kp = csvp(L + r"\C16\C16-T3\kr_prefix.csv")
def K(src, pre, col): return float(kp[(kp["source"].str.startswith(src)) & (kp["prefix"] == pre)][col].iloc[0])
nall, n10, n20 = K("meta KR", "ALL", "n"), K("meta KR", "10", "n"), K("meta KR", "20", "n")
ln494 = next(i for i, s in enumerate(MS, 1) if s.startswith("| C-COUNTRY-SHARE |"))
rec("E-C16T3-02", ln494, grab(ln494, r"KR (\d+\.\d)%\(([\d,]+); 실용신안 ([\d,]+) 포함\)", 3), [r_(100 * nall / 15141159, 1), str(int(nall)), str(int(n20))], r"C16-T3\kr_prefix.csv")
la, l10 = K("ntis_patent_flag(LINK", "ALL", "n_linked"), K("ntis_patent_flag(LINK", "10", "n_linked")
ra, r10 = K("ntis_patent_flag(LINK", "ALL", "n_linked_registered"), K("ntis_patent_flag(LINK", "10", "n_linked_registered")
ga, g10 = K("ntis_patent_flag(LINK", "ALL", "n_registered"), K("ntis_patent_flag(LINK", "10", "n_registered")
ln516 = next(i for i, s in enumerate(MS, 1) if s.startswith("| LINK-NTIS |"))
g = grab(ln516, r"특허 ([\d,]+)·실용신안 ([\d,]+)\)의 NTIS 성과 연결 (\d+\.\d+)%\(([\d,]+)/[\d,]+; 특허만 (\d+\.\d+)%\), 등록 (\d+\.\d+)%\(([\d,]+)/[\d,]+; 특허만 (\d+\.\d+)%\)", 8)
rec("E-C16T3-01", ln516, g, [str(int(n10)), str(int(n20)), r_(100 * la / nall, 2), str(int(la)), r_(100 * l10 / n10, 2), r_(100 * ra / ga, 2), str(int(ra)), r_(100 * r10 / g10, 2)], r"C16-T3\kr_prefix.csv")
rec("표3-③", 96, grab(96, r"KR 출원의 (\d+\.\d+)%, 등록의 (\d+\.\d+)%", 2), [r_(100 * la / nall, 2), r_(100 * ra / ga, 2)], r"C16-T3\kr_prefix.csv")
# 7 PKR-IPC
pk = csvp(L + r"\C14\C14-T2\pkr_ipc_altA.csv").set_index("cc")
for cc in ["US", "JP", "CN", "EP"]:
    ln = next(i for i, s in enumerate(MS, 1) if s.startswith("| C-AGREE-" + cc + " |"))
    rec("E-C14T2-03:" + cc, ln, grab(ln, r"진단용 PKR-IPC (\d+\.\d)%"), [r_(float(pk.loc[cc, "대안A_REG"]), 1)], r"C14-T2\pkr_ipc_altA.csv 대안A_REG")
# 8 656,360 — 다른 질의(sum case)
con = sqlite3.connect("file:" + REG.replace("\\", "/") + "?mode=ro", uri=True)
tot, p10, p20, nul = con.execute("select count(*), sum(rgst_no like '10%'), sum(rgst_no like '20%'), sum(rgst_no is null) from patent").fetchone(); con.close()
ln503 = next(i for i, s in enumerate(MS, 1) if s.startswith("| DATA-KR-REG |"))
rec("E-C15T4-07b", ln503, grab(ln503, r"KR 등록 ([\d,]+)건\(특허 ([\d,]+)·실용신안 ([\d,]+)\)", 3), [str(tot), str(p10), str(p20)], r"registry_full.sqlite(mode=ro, like 질의)", f"null {nul}")
rec("E-C15T4-07", 79, grab(79, r"등록특허·실용신안 ([\d,]+)건"), [str(tot)], "registry_full.sqlite")
# 9 535 C16-T1-VIS
v16 = csvp(L + r"\C16\C16-T1\variants.csv").set_index("variant")
ln = next(i for i, s in enumerate(MS, 1) if s.startswith("| C16-T1-VIS |"))
g = grab(ln, r"skill (0\.\d+) \[(0\.\d+), (0\.\d+)\]\(C 법인 행 재계산\)·(0\.\d+)\(공식 행 점수 고정\)", 4)
rec("E-C16T1-B1", ln, g, [r_(v16.loc["S1full_La", "skill"], 3), r_(v16.loc["S1full_La", "ci_lo"], 3), r_(v16.loc["S1full_La", "ci_hi"], 3), r_(v16.loc["S1fixed_La", "skill"], 3)], r"C16-T1\variants.csv")
# 10 536 C17-T1-NEWC
v1o = jl(L + r"\C17\C17-T1\verify_1\v1_out.json")
ln = next(i for i, s in enumerate(MS, 1) if s.startswith("| C17-T1-NEWC |"))
g = grab(ln, r"N1 Δ \+(0\.\d+) \[\+(0\.\d+), \+(0\.\d+)\]\(무작위 동수 제외 \+(0\.\d+)", 4)
rec("E-C17T1-B1", ln, g, [r_(v.loc["N1re", "delta_skill"], 4), r_(v.loc["N1re", "dci_lo"], 4), r_(v.loc["N1re", "dci_hi"], 4), r_(v1o["rand_drop_same_nfirms_delta"]["mean"], 4)], r"C17-T1\variants.csv, verify_1\v1_out.json")
# 11 537 C17-T2-ALIAS
ln = next(i for i, s in enumerate(MS, 1) if s.startswith("| C17-T2-ALIAS |"))
g = grab(ln, r"오매칭 (\d+\.\d)% \[(\d+\.\d), (\d+\.\d)\]", 3)
ci = s2["main_E2"]["boot_ci"]
rec("E-C17T2-B1", ln, g, [r_(100 * s2["main_E2"]["e"], 1), r_(100 * ci[0], 1), r_(100 * ci[2], 1)], r"C17-T2\cache\s2_out.json")
# 12 538·217 C17-T3-NEWIPC
ln = next(i for i, s in enumerate(MS, 1) if s.startswith("| C17-T3-NEWIPC |"))
g = grab(ln, r"셀 (\d+)\((\d+\.\d)%, 양성 (\d+)\)", 3)
n0, n1 = float(t3f.loc["V0", "n"]), float(t3f.loc["V1", "n"]); p0, p1 = float(t3f.loc["V0", "pos"]), float(t3f.loc["V1", "pos"])
rec("E-C17T3-B1", ln, g, [str(int(n0 - n1)), r_(100 * (n0 - n1) / n0, 1), str(int(p0 - p1))], r"C17-T3\t3_newipc_sens.csv", f"분모 V0 n={n0:.0f}")
rec("표6", 217, grab(217, r"\| 신설 IPC 셀 제외 \| (0\.\d+) \|"), [r_(t3f.loc["V1", "skill"], 3)], r"C17-T3 V1")
# 13 388·655
hz = csvp(L + r"\C15\C15-T1\cache\s4_horizon.csv")
H = lambda h, gname: float(hz[(hz["h"] == str(h)) & (hz["group"] == gname)]["regdate"].iloc[0])
lap = jl(L + r"\C02\C02-T1\_tab_e_lapse.json"); st = lap["standardized"][0]
g = grab(388, r"5년 내 포기 (\d+\.\d)%\(비기원 (\d+\.\d)%; 유형 표준화 격차 (\d\.\d)–(\d\.\d)%p", 4)
rec("E-C16T4-01", 388, g, [r_(100 * H(5, "정부R&D 기원"), 1), r_(100 * H(5, "비기원"), 1), r_(st["gap_pp_govmix"], 1), r_(st["gap_pp_nongovmix"], 1)], r"C15-T1 s4_horizon, C02-T1 _tab_e_lapse.json",
    "10년 역전 " + str(H(10, "정부R&D 기원") > H(10, "비기원")))
ln655 = next(i for i, s in enumerate(MS, 1) if s.startswith("| NTIS-T3-SUM |"))
g = grab(ln655, r"이전 (\d+\.\d)% 대 (\d+\.\d)%; 7년 내 (\d+\.\d)% 대 (\d+\.\d)%, 10년 내\(등록 2007–15\) (\d+\.\d)% 대 (\d+\.\d)%", 6)
rec("E-C15T1-02", ln655, g, [r_(100 * lap["recompute"]["transfer5_gov"], 1), r_(100 * lap["recompute"]["transfer5_nongov"], 1), r_(100 * H(7, "정부R&D 기원"), 1), r_(100 * H(7, "비기원"), 1), r_(100 * H(10, "정부R&D 기원"), 1), r_(100 * H(10, "비기원"), 1)], r"C02-T1, C15-T1")
# 14 143
s10 = csvp(L + r"\C10\C10-T1\sens_results.csv").set_index("variant")
s14 = csvp(L + r"\C14\C14-T3\sens_results.csv").set_index("variant")
ov = csvp(L + r"\C02\C02-T2\official_vs_provided.csv"); ov["year"] = ov["year"].astype(int)
kr = ov[(ov["country"] == "KR") & ov["year"].between(2012, 2023)]["ratio_pat"].astype(float)
cn = ov[(ov["country"] == "CN") & ov["year"].between(2018, 2022)]["ratio_t3year"].replace("", "nan").astype(float)
cn_inv = ov[(ov["country"] == "CN") & ov["year"].between(2018, 2022)]["ratio_inv_rgst"].replace("", "nan").astype(float)
g = grab(143, r"하락 (0\.\d+) 이하\), .*?영향 (0\.\d+) 이하", 2)
rec("143행", 143, g, [r_(max(-float(s10.loc["S4", "d_skill"]), -float(s10.loc["S5", "d_skill"])), 4), r_(abs(float(s14.loc["UPPER", "final_d_skill"])), 3)], r"C10-T1 S4·S5, C14-T3 UPPER(판정 지표, 부호 없는 크기)")
g = grab(143, r"KR 등록 2012–23년 (0\.\d+)–(0\.\d+), CN 2018–22년 (0\.\d+)–(0\.\d+)", 4)
rec("143행b·CN(추가)", 143, g, [r_(kr.min(), 2), r_(kr.max(), 2), r_(cn.min(), 2), r_(cn.max(), 2)], r"C02-T2\official_vs_provided.csv (KR ratio_pat, CN ratio_t3year)",
    f"CN ratio_t3year {cn.min():.4f}–{cn.max():.4f}; 발명 등록 기준 ratio_inv_rgst는 {cn_inv.min():.4f}–{cn_inv.max():.4f}(2021–22 등록 결손) → 원고 CN 0.20–0.21은 T3 연도(발명+실용) 기준")
# 15 168·482
v5 = jl(L + r"\C11\C11-T2\verify_5\v5_out.json")["range_168"]; vf = jl(L + r"\C11\C11-T2\verify_5\v5_frac.json")
g = grab(168, r"한국 RTA 순위상관은 (0\.\d+)–(0\.\d+)\(동률 규칙에 따라 최대 (0\.\d+)\)", 3)
rec("168행", 168, g, [r_(v5["REG_base"]["min_raw"], 2), r_(v5["REG_base"]["max_raw"], 2), r_(max(x["max_raw"] for x in v5.values()), 2)], r"C11-T2\verify_5\v5_out.json")
ln482 = next(i for i, s in enumerate(MS, 1) if s.startswith("| C-LAND-free_F |"))
g = grab(ln482, r"ρ\(제공 대비\) (0\.\d+)\(최빈 동률 규칙 의존: 표 동률만 반대 (0\.\d+), 배정까지 반대 (0\.\d+)", 3)
rec("482행", ln482, g, [r_(vf["asc_free_F"]["rho"], 3), r_(vf["desc_free_F"]["rho"], 3), r_(v5["REG_tie_desc_full"]["max_raw"], 3)], r"C11-T2\verify_5\v5_frac.json")
# 16 시드 10묶음
sb = csvp(L + r"\C15\C15-T2\seed_bundles.csv")
t1 = sb[sb["track"] == "T1"]; t3s = sb[sb["track"] == "T3"]
f = lambda d, c: d[c].replace("", "nan").astype(float).dropna()
ln = next(i for i, s in enumerate(MS, 1) if "다른 학습 시드 10묶음" in s and "0.472" in s)
g = grab(ln, r"(0\.472)–(0\.475), 하한 (0\.453)–(0\.456)", 4)
rec("E-C15T2-02", ln, g, [r_(f(t1, "skill").min(), 3), r_(f(t1, "skill").max(), 3), r_(f(t1, "ci_lo").min(), 3), r_(f(t1, "ci_lo").max(), 3)], r"C15-T2\seed_bundles.csv", f"T1 n={len(t1)}")
# 17 142 (미반영 0자 제안 대상)
U = jl(L + r"\C03\C03-T3\_s8_unseen.json"); uv = [U[k]["unseen_min_pub_rgst"] for k in U]
g = grab(142, r"약 (\d+)–(\d+)%가 원점에 미공개", 2)
rec("E-C09T2-04(현행)", 142, g, [r_(100 * min(uv), 0), r_(100 * max(uv), 0)], r"C03-T3\_s8_unseen.json", f"{100*min(uv):.2f}–{100*max(uv):.2f}% → 현행 46은 '약' 허용(±1), 제안 47–48이 사사오입값", approx=True)
# 18 518 KOSIS
oc = csvp(L + r"\C07\C07-T1\official_compare.csv")
ok_ = (oc["status"] == "일치(≤0.5%)").sum(); comp = ((oc["status"] == "일치(≤0.5%)") | oc["status"].str.startswith("불일치")).sum()
rec("E-C07T1-08(제안값)", 518, ["5925", "5925"], [str(ok_), str(comp)], r"C07-T1\official_compare.csv", "제안 문안 값(원고 현행 49/50은 2008년 값)")
# 19 515 소멸등록, NTIS 대조
ev = jl(L + r"\C15\C15-T1\cache\s2_out.json")["event_counts_by_prefix"]
rec("E-C15T1-03(제안값)", 515, ["349507", "293169", "20112"], [str(sum(ev[k]["10"] + ev[k]["20"] for k in ("NP", "EXP", "EXT_UNK"))), str(ev["NP"]["10"]), str(ev["EXP"]["10"])], r"C15-T1\cache\s2_out.json")
nc = csvp(L + r"\C17\C17-T3\cache\ntis_count_check.csv"); rr = nc["ratio_n"].replace("", "nan").astype(float).dropna()
rec("E-C17T3-B2(제안값)", 513, ["0.99993", "1.00000"], [r_(rr.min(), 5), r_(rr.max(), 5)], r"C17-T3\cache\ntis_count_check.csv")
# 20 T4 강화 하한 −0.003 (s3는 재계산 안 함 → 원천 md 표와 C13-T2 seed_bundles로 직접)
md = open(MC + r"\T4_특허포기예측\VERIFY_R5\robustness_table.md", encoding="utf-8").read()
m = re.search(r"로지스틱 13변수\*\* \| \*\*(0\.\d+)\*\* \| \*\*(0\.\d+)\*\* \| \*\*\[([−\-]\d\.\d+), (0\.\d+)\]\*\*", md)
ln533 = next(i for i, s in enumerate(MS, 1) if s.startswith("| OFF-T4-STRONG |"))
g = grab(ln533, r"\((0\.\d+)\) 대비 skill (0\.\d+) \[([−\-]0\.\d+), (0\.\d+)\]", 4)
rec("OFF-T4-STRONG(추가)", ln533, g, [r_(m.group(1), 3), m.group(2), m.group(3), m.group(4)], r"(MC)\T4_특허포기예측\VERIFY_R5\robustness_table.md")
sbt = csvp(L + r"\C13\C13-T2\seed_bundles.csv"); t4 = sbt[sbt["track"] == "T4"]["strong_ci_lo"].replace("", "nan").astype(float).dropna()
g130 = grab(130, r"CI 하한이 0에 닿는 T4\(([−\-]0\.\d+)\)"); g409 = grab(409, r"CI 하한이 0에 닿고\(([−\-]0\.\d+)\)")
rec("E-C13T2-02(130·409, 추가)", 130, g130 + g409, [m.group(3), m.group(3)], r"robustness_table.md + C13-T2 seed_bundles",
    f"C13-T2 T4 시드 묶음 강화 대비 하한 {t4.min():.4f}–{t4.max():.4f} (n={len(t4)}, 모두 <0: {bool((t4 < 0).all())})")
# 21 본문 토큰 스캔 미대응 중 원천 확인(추가)
c4 = jl(L + r"\C04\C04-T3\s3_results.json"); c4b = jl(L + r"\C04\C04-T3\s2_results.json"); c5j = jl(L + r"\C05\C05-T1\s2_results.json")
g = grab(219, r"적중 (\d+)셀의 (\d+)%는 직전 KR 1건 이하 셀이고 (\d+)%는 증가 3건 이하이며, 부상의 (\d+)%", 4)
h = c4["top504"]["hits"]
rec("219행(추가)", 219, g, [str(h), r_(100 * c4["top504"]["hits_krW_le1"] / h, 0), r_(100 * c4["top504"]["hits_D_le3"] / h, 0), r_(100 * c4["sab_k1"]["rate"], 0)], r"C04-T3\s3_results.json")
rec("409행 45%(추가)", 409, grab(409, r"부상 순증가의 (\d+)%가 H10"), [r_(100 * c5j["final_impact"]["A1"]["netD_share"], 0)], r"C05-T1\s2_results.json final_impact.A1.netD_share")
n6 = jl(L + r"\C06\C06-T3\n6_rescore_blank.json")["scenarios"]
fb = jl(L + r"\C06\C06-T3\verify_0\v1_ntis_fallback_out.json")
mv = [x["dskill"] for k, x in n6.items() if k.startswith("M") and k.endswith("drop=True")] + [x["dskill"] for k, x in fb.items() if isinstance(x, dict) and k.startswith("M+NTIS") and k.endswith("drop=True")]
rec("410행 N6(추가)", 410, grab(410, r"N6\(미등록 포함 동명 복원\) \+(0\.\d+)–(0\.\d+)", 2), [r_(min(mv), 3), r_(max(mv), 3)],
    r"C06-T3\n6_rescore_blank.json(M·M_sido) + verify_0\v1_ntis_fallback_out.json(M+NTIS지역·strict), 위험집합 제외판",
    f"{len(mv)}개 명세 {min(mv):.4f}–{max(mv):.4f}")
c4t1 = csvp(L + r"\C04\C04-T1\scenario_rescore.csv")
print("C04-T1 scenario columns:", list(c4t1.columns)[:12])
json.dump(OUT, open(W + r"\verify\v2_numbers_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.DataFrame(OUT).to_csv(W + r"\verify\v2_numbers_indep.csv", index=False, encoding="utf-8-sig")
print({k: sum(1 for o in OUT if o["독립판정"] == k) for k in ("일치", "불일치")})

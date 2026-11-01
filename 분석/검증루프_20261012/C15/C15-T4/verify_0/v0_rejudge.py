# C15-T4 verify_0: 1차 판정 스크립트(s2)·FINDINGS 요약을 쓰지 않고, 원고 v16과 원 결과 파일을 직접 열어 독립 재판정한다.
# 대상: 1차 비지지 7개 + 지지 무작위 15개(seed 2026101504, sample_ids.json). 반박 시도: R1(같은 절·표 주·4.4 단서 탐색), R2(원천에서 강도를 받치는 다른 값 탐색).
import csv, json, re, os, sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = B + r"\모델경쟁_20260927"; VL = B + r"\검증루프_20261012"; W = VL + r"\C15\C15-T4"
M = open(B + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().split("\n")
SUB = [l for l in M[:444]]  # 제출본 범위(부록 전)
def line(n): return M[n - 1]
def txt(p): return open(p, encoding="utf-8", errors="replace").read()
def sec_lines(a, b): return "\n".join(M[a - 1:b])
smp = json.load(open(W + r"\verify_0\sample_ids.json", encoding="utf-8"))
R, V = [], {}

def rec(uid, first, verdict, vals, r1, r2, note=""):
    V[uid] = vals
    R.append(dict(uid=uid, first=first, verify=verdict, flipped=(first != verdict), R1=r1, R2=r2, note=note))

# ---------- 비지지 7개 ----------
# U007 2.1 '처음부터 잠가 정답을 열어 보지 않았고'
r2 = txt(MC + r"\R2_결과_20260928.md"); p3 = txt(MC + r"\_shared\PROTOCOL_R3.md"); sp = txt(MC + r"\_shared\SPLITS.md")
t3_r2 = re.search(r"t0=2019\((\d{4})–(\d{2})\), t0=2020\((\d{4})–(\d{2})\)", r2)
t1_r2 = re.search(r"c=2017\(진입 (\d{4})–(\d{2})\)", r2)
t1_r3 = re.search(r"c=2019 → 진입 (\d{4})–(\d{4})", p3)
t3_r1 = re.search(r"t0=2018: 성장 (\d{4})–(\d{4})", sp)
seen_note = "보기만 하고 학습·튜닝에 쓰지 않는다" in p3
t1_overlap = sorted(set(range(int(t1_r2.group(1)), 2000 + int(t1_r2.group(2)) + 1)) & set(range(int(t1_r3.group(1)), int(t1_r3.group(2)) + 1)))
disclosed = [k for k in ("0.392 [0.348", "0.418 [0.402", "출원연도판", "2라운드", "앞 라운드", "이전 라운드") if any(k in l for l in SUB)]
r1_hit = [k for k in ("겹", "앞 라운드", "이전 라운드") if k in sec_lines(105, 113) + line(409)]
rec("U007", "과장", "과장", dict(T3_R1=t3_r1.groups(), T3_R2=t3_r2.groups(), T1_R2=t1_r2.groups(), T1_R3=t1_r3.groups(), T1_overlap_years=t1_overlap,
     protocol_R3_says_seen=seen_note, prior_finals_disclosed_in_submission=disclosed),
    f"R1: 2.1(105–113행)·4.4-3에 앞 라운드 겹침 단서 {r1_hit or '없음'}; 제출본 안 앞 라운드 값 표기 {disclosed or '없음'}",
    "R2: '한 번만 채점'은 동결 모형 단위로는 맞음(강도 일부 받침) → '모순' 아닌 '과장' 유지. PROTOCOL_R3가 2라운드 결과를 '보기만' 했다고 적어 '열어 보지 않았다'와 충돌",
    "T1 겹침 연도 " + ",".join(map(str, t1_overlap)))

# U019 2.5 연도별 건수 비율 안정
ov = list(csv.DictReader(open(VL + r"\C02\C02-T2\official_vs_provided.csv", encoding="utf-8-sig")))
def rng(cc, col, ys):
    v = [float(r[col]) for r in ov if r["country"] == cc and int(r["year"]) in ys and r[col] not in ("", None)]
    return (round(min(v), 3), round(max(v), 3)) if v else None
vals = dict(KR_pat_2012_23=rng("KR", "ratio_pat", range(2012, 2024)), KR_pat_2024=rng("KR", "ratio_pat", [2024]),
            CN_t3year_2018_22=rng("CN", "ratio_t3year", range(2018, 2023)), CN_invpub_2012_14=rng("CN", "ratio_inv_pub", range(2012, 2015)),
            CN_invpub_2015_22=rng("CN", "ratio_inv_pub", range(2015, 2023)), CN_invrgst_2021_24=rng("CN", "ratio_inv_rgst", [2021, 2022, 2024]))
r1 = "4.4에 CN 초기 연도 단서 " + ("있음" if re.search(r"CN 20(1[0-4]|0\d)", sec_lines(405, 411)) else "없음") + "; 143행 괄호에 기간 표기 " + ("있음" if re.search(r"20\d\d–\d\d년?\s*0\.2", line(143)) else "없음")
rec("U019", "단서 누락", "단서 누락", vals, r1, "R2: 원고 값(0.25–0.26, 0.20–0.21)은 KR 2012–23·CN 2018–22 창에서만 재현. CN 발명 공개연도 2012–14는 0.10–0.16으로 '안정' 반례")

# U020·U039 T1 권리목록 출처
c14 = txt(VL + r"\C14\C14-T3\summary.txt"); t1f = txt(MC + r"\T1_산업대기실\R3\FINAL_REPORT_R3.md")
gain = re.search(r"범위밖 의존 42열 전체 gain ([\d.]+)%", c14); top = re.search(r"1–4위는 ([^.\n]+)", t1f)
tab3 = sec_lines(84, 99)
v = dict(gain_42cols=gain.group(1) if gain else None, t1_top_features=top.group(1) if top else None,
         table3_has_rightlist=("권리목록" in tab3), line229_has_rightlist=("권리목록" in line(229)), line251_has_rightlist=("권리목록" in line(251)),
         line143_mentions_T1_rightlist=("T1 권리목록" in line(143)), line410_mentions_rightlist=("사업자 권리목록" in line(410)), table13_has=("사업자 권리목록" in line(426)))
rec("U039", "단서 누락", "단서 누락", v, "R1: 같은 절(3.3.1)·그림 6 캡션에 없음. 143행(2.5)·410행(4.4-4)은 권리목록이 2026 스냅샷이라고만 적고 T1 예측 피처라는 점·비중은 없음 → 단서로 보기 어려움",
    "R2: T1 트랙 문서(FINAL_REPORT 140행)는 외부데이터를 디자인·실용신안·물가·산업연관표로만 부름 — 원고는 트랙 문서 분류를 따른 것. 그러나 권리목록은 KIPRIS Plus 외부 API 자료이고 표 13이 출처로 적음 → 누락 유지")
rec("U020", "단서 누락", "단서 누락", v, "R1: 캡션이 '표 3의 외부 데이터'로 범위를 정했는데 표 3·각주 ¹에 권리목록 없음(표 13에는 있음)",
    "R2: 그림 3은 9개 트랙 성적 그림이라 출처를 '표 13'으로 읽을 여지 — 캡션 문자는 '표 3'이므로 누락 유지(경미)")

# U031 3.1.3 코드 안 분산
vd = {r["지표"]: float(r["그룹안_분산비중_34"]) for r in csv.DictReader(open(MC + r"\_shared_report\fusion_advantage\split34\t1_variance_decomposition.csv", encoding="utf-8-sig"))}
v = {k: vd[k] for k in ("g_KR", "g_US", "g_JP", "g_EP", "g_CN", "g_5국")}
same = all(abs(v[k] - v["g_KR"]) < 0.05 for k in ("g_US", "g_JP", "g_EP"))
rec("U031", "단서 누락", "단서 누락" if same else "지지", v, "R1: 같은 글머리·그림 5 캡션·4.4에 US·JP·EP 비교 없음",
    "R2: '한국 지표는 그렇지 않다'는 문자 그대로 참(0.561, 무작위 기대 0.698 대비 p 0.115) → 과장 아님, 단서 누락 유지")

# U042 N9 AUC 우위
n9 = json.load(open(MC + r"\N9_정부RnD과제_특허성과예측\final_result_r6.json", encoding="utf-8"))
v = dict(model=n9["auc"], pilot_orig=n9["auc_pilot_orig"], strong_orig=n9["auc_strong_orig"], orig_champion=n9["secondary_orig_champion"]["auc"])
rec("U042", "단서 누락", "단서 누락" if v["orig_champion"] > v["model"] else "지지", v,
    "R1: 265행·4.4-3에 원판 챔피언 AUC 없음(4.4-3은 상위 1%만)", "R2: 문장이 비교 대상(파일럿 원판·강한 기준선 원판)으로 한정해 읽히면 참 → 과장 아님, 단서 누락")

# U086 3.6.5 종합 T3 경계
pure = json.load(open(VL + r"\C11\C11-T2\res\s5_summary_mg__pure.json", encoding="utf-8"))
s = json.dumps(pure, ensure_ascii=False)
t10 = line(355)
v = dict(table10_row=t10, pure_json_head=s[:300])
rec("U086", "단서 누락", "단서 누락" if "경계" in t10 and "경계" not in line(362).split("공개 배정이 낫다")[0] else "지지", v,
    "R1: 같은 절 표 10(355행)에 '같다(경계)' — 표에는 단서가 있으나 종합 문장은 '같고'로 단정. C13 기준(336행 같은 유형 '단서 누락')과 같게 유지",
    "R2: 대안 A(원고 채택) CI는 0 포함 → '같다'는 주 정의에서 참. 강도 차이만 → 단서 누락(보수 방향)")

# ---------- 지지 무작위 15개 ----------
t3row = sec_lines(86, 97)
rec("U001", "지지", "지지" if all(k in t3row for k in ("KOSIS", "Comtrade", "산업연관표", "국민연금", "DART", "NTIS", "등록원부")) else "단서 누락", dict(table3_ok=True), "", "")
defs = dict(N4="국세청" in line(289), N5="ETSI" in line(99), N7="NTIS 과제를 처음" in line(275), N6_pop="NTIS 과제 수행기업" in line(267), N9_pop="과제연도 2022년 과제" in line(259))
erra = txt(VL + r"\ERRATA.txt"); orig = "① 모든 트랙은 모집단이나 결과를 제공 데이터로 정의한다(N6·N9 모집단은 NTIS, N4·N5·N7 결과는 외부 사건; 등록원부도"
reg = sqlite3.connect("file:" + B.replace("\\분석", "") + r"\수집파이프라인\registry_full.sqlite?mode=ro", uri=True)
um = dict(reg.execute("select substr(rgst_no,1,2), count(*) from patent group by 1").fetchall())
rec("U003", "지지", "지지" if all(defs.values()) and orig in erra and orig in line(79) else "단서 누락", dict(defs=defs, errata_text_match=(orig in erra and orig in line(79)), registry_prefix=um),
    "", "부수: 656,360 = 특허 %d + 실용신안 %d → '등록특허' 표기 경미 오류(원칙 판정과 별개)" % (um.get("10", 0), um.get("20", 0)))
rec("U009", "지지", "지지" if all("보수적 skill" in sec_lines(a, b) for a, b in ((211, 217), (235, 240), (279, 284))) else "단서 누락", {}, "", "")
m2 = json.load(open(MC + r"\_shared_report\fusion_advantage\map5\m2_checks.json", encoding="utf-8")); m7 = txt(MC + r"\_shared_report\fusion_advantage\map5\m7_summary.json")
ok16 = m2["coverage_flagged_cells_n"] == 175 and m2["coverage_cells_n"] == 3255 and len(m2["coverage_flagged_total"]) == 5 and ("63886" in m7 or "63,886" in m7) and round(json.loads(m7)["jp_2021_ksic_rank_raw_vs_corr_spearman"], 3) == 0.998
rec("U016", "지지", "지지" if ok16 else "근거 없음", dict(flagged=m2["coverage_flagged_cells_n"], cells=m2["coverage_cells_n"], m7_has_63886=("63886" in m7), m7_rank_rho=json.loads(m7)["jp_2021_ksic_rank_raw_vs_corr_spearman"]), "", "")
f2 = txt(VL + r"\FINDINGS.txt").split("\n")[1]
rec("U017", "지지", "지지" if ("보정 없는" in f2 or "원계열" in f2) else "근거 없음", dict(findings2=f2[:160]), "", "")
s10 = {r["variant"]: float(r["d_skill"]) for r in csv.DictReader(open(VL + r"\C10\C10-T1\sens_results.csv", encoding="utf-8-sig"))}
s14 = {r["variant"]: float(r["final_d_skill"]) for r in csv.DictReader(open(VL + r"\C14\C14-T3\sens_results.csv", encoding="utf-8-sig"))}
s3m = sum(v for k, v in s14.items() if k.startswith("S3_")) / max(1, sum(1 for k in s14 if k.startswith("S3_")))
bs = txt(VL + r"\C12\C12-T3\b_sens.csv")
rec("U018", "지지", "지지" if (min(s10.get("S4", 0), s10.get("S5", 0)) >= -0.00225 and s3m >= -0.0075) else "단서 누락",
    dict(T1_S4=s10.get("S4"), T1_S5=s10.get("S5"), T1_rightlist_S3_mean=round(s3m, 5), b_sens_head=bs[:200]), "", "S3 시드 최대 −0.0084는 0.007 초과 — '영향 0.007 이하'는 시드 평균 기준(C14 권고 문구)")
s5 = txt(VL + r"\C14\C14-T3\cache\s5_out.json")
rec("U037", "지지", "지지" if ("3260" in s5 or "3,260" in s5) and "2019년까지 출원한 특허 3건 이상" in line(229) else "단서 누락", dict(s5_has_3260=("3260" in s5)), "", "")
pop = list(csv.DictReader(open(VL + r"\C02\C02-T1\_tab_e_n6_population.csv", encoding="utf-8-sig")))
ro = {r["t0"]: float(r["rate"]) for r in pop if r["version"] == "official" and r["rate"]}
rc = {r["t0"]: float(r["rate"]) for r in pop if r["version"] == "corrected" and r["rate"]}
dmax = max(abs(rc[t] - ro[t]) for t in ro if t in rc) * 100
rec("U043", "지지", "지지" if dmax <= 0.1 else "모순", dict(max_rate_diff_pp=round(dmax, 4), origins=len([t for t in ro if t in rc])), "",
    "run1은 verify_note만 읽어 값 확인 없이 '지지'로 둔 결함 → run2에서 official 대 corrected 개발 원점 양성률 차를 직접 계산")
rm = txt(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_main.csv")
rec("U060", "지지", "지지" if ("-0.0358" in rm or "−0.0358" in rm or "-0.035" in rm) and "0.0809" in rm else "근거 없음", dict(has_m0358=("-0.0358" in rm), has_0809=("0.0809" in rm)), "", "")
se = list(csv.DictReader(open(VL + r"\C14\C14-T1\se_compare.csv", encoding="utf-8-sig")))
ind = [r for r in se if ("목표KSIC" in r["structure"] or "업종" in r["structure"]) and r["origin"] == "pooled" and "CGM" in r["estimator"] and r["pair"] == "P50-FREE50"]
anyind = [r for r in se if "업종" in r["structure"] and "CGM" in r["estimator"]]
rec("U075", "지지", "지지" if any(float(r["lo"]) < 0 < float(r["hi"]) for r in (ind or anyind)) else "단서 누락",
    dict(industry_CGM_rows=[(r["origin"], r["lo"], r["hi"]) for r in (ind or anyind)][:8]), "", "")
rp = txt(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_prep.json") + txt(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_tests.json")
rec("U093", "지지", "지지" if ("0.3032" in rp or "0.30317" in rp or "0.3031" in rp) else "근거 없음", dict(has_03032=("0.3032" in rp or "0.3031" in rp)), "", "")
fr5 = json.load(open(VL + r"\C11\C11-T2\verify_5\v5_frac.json", encoding="utf-8"))
rec("U096", "지지", "지지" if abs(fr5["desc_free_F"]["rho"] - 0.473) < 0.0006 else "모순", dict(desc_free_F=fr5["desc_free_F"]["rho"]), "", "본문 168 '최대 0.48'의 출처값 0.4755는 refute_result.txt — 부록 동기화 필요(판정과 별개)")
pk = {r["cc"]: r["대안A_REG"] for r in csv.DictReader(open(VL + r"\C14\C14-T2\pkr_ipc_altA.csv", encoding="utf-8-sig"))}
rec("U102", "지지", "지지" if abs(float(pk["EP"]) - 72.8) < 0.06 and "72.8%" in line(501) else "모순", dict(EP=pk["EP"]), "", "")
m4 = json.load(open(MC + r"\_shared_report\fusion_advantage\map5\m4_summary.json", encoding="utf-8"))
rec("U117", "지지", "지지" if abs(float(m4["quad_P2_vs_P4_same_share"]) - 0.854) < 1e-6 else "모순", dict(persist=m4["quad_P2_vs_P4_same_share"]), "", "")
ph = txt(MC + r"\_shared_report\label_origin_R10\res_posthoc2.json")
rec("U129", "지지", "지지" if ("0.301" in ph and "0.304" in ph) or ("0.548" in ph) else "근거 없음", dict(has_values=("0.301" in ph, "0.304" in ph, "0.616" in ph, "0.628" in ph)), "", "")

os.makedirs(W + r"\verify_0", exist_ok=True)
with open(W + r"\verify_0\verify_result.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(R[0].keys())); w.writeheader(); [w.writerow(x) for x in R]
json.dump(V, open(W + r"\verify_0\v0_values.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
sup = [x for x in R if x["first"] == "지지"]; non = [x for x in R if x["first"] != "지지"]
print("support sample flips", sum(x["flipped"] for x in sup), "/", len(sup), "| nonsupport flips", sum(x["flipped"] for x in non), "/", len(non))
for x in R: print(x["uid"], x["first"], "->", x["verify"], "|", json.dumps(V[x["uid"]], ensure_ascii=False, default=str)[:260])

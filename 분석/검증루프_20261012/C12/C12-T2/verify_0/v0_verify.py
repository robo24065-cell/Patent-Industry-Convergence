# C12-T2 verify_0: 작업 스크립트(s7: pandas + 작업자 res 파일)와 다른 코드 경로로 반박 시도
#  - 판독: csv 모듈/json 모듈만(pandas 미사용), 반올림은 float 문자열 포맷이 아니라 정수 연산(round-half-up)으로 직접 구현
#  - 원천: 가능하면 작업자 res 대신 각 바퀴의 독립 검증 산출물(verify_*)이나 원점별 AUC에서 다시 합성
#  (1) '부분'·'낡은 표기'·'오류' 판정 전부, (2) '정확' 하위 블록 중 무작위 15개(시드 20261013)
import sys, json, csv, random, re, math, os
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = AN + r"\검증루프_20261012"; MC = AN + r"\모델경쟁_20260927"; W = L + r"\C12\C12-T2"
ms = open(AN + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().split("\n")
def J(p): return json.load(open(p, encoding="utf-8"))
def rows(p):
    with open(p, encoding="utf-8-sig", newline="") as fh: return list(csv.DictReader(fh))
def rhu(x, d):  # round-half-up via integer arithmetic on a 12-digit scaled value
    m = 10 ** d; y = abs(x) * m
    q = int(math.floor(y + 0.5 + 1e-12))
    s = ("%d" % (q // m)) + (("." + ("%0" + str(d) + "d") % (q % m)) if d else "")
    return ("-" if x < 0 and q else "") + s
out = []
def rec(kind, item, line, expect, got, ok, src, note=""):
    out.append(dict(kind=kind, item=item, line=line, manuscript=expect, independent=got, agree=ok, source=src, note=note))
# ---------- 독립 원천 ----------
v1 = J(L + r"\C10\C10-T3\verify_0\v1_out.json")          # C10-T3 검증자 독립 코드
v4b = J(L + r"\C10\C10-T3\verify_4\v4b_out.json")        # T3 원점별 AUC(검증자 재학습)
v1p = J(L + r"\C10\C10-T2\verify_1\v1_point.json")       # T1 원점별 AUC(검증자)
sc = J(L + r"\C10\C10-T2\verify_1\v1_boot_sc.json")
c11v1 = J(L + r"\C11\C11-T2\verify_1\v1_out.json")
c11r2 = J(L + r"\C11\C11-T2\verify_0\r2_c.json")
c11auc = J(L + r"\C11\C11-T2\verify_1\v1b_auc.json")
v3rows = rows(L + r"\C11\C11-T2\verify_3\v3_rows.csv")
sensj = J(L + r"\C10\C10-T1\sens_results.json")
c1 = {(r["window"], r["cc"]): r for r in rows(MC + r"\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv")}
agree_rows = rows(L + r"\C10\C10-T3\res\s2_agree.csv")
facts = {}
for fn in ["facts.csv", "facts_supplement.csv"]:
    for r in rows(AN + r"\보고서_20261012\facts\\" + fn):
        k = list(r.values())[0]; facts.setdefault(k, []).append((fn, r))
# ---------- (1) 부분·낡은 표기 판정 반박 시도 ----------
# B02/B03/B14 부분(c): 부록 498·499·501·534행이 R8 결함 연결값(=c1_label_agreement 저장값)으로 남았나
fr = v1["free_recall"]
for cc, ln in [("US", 498), ("JP", 499), ("EP", 501)]:
    stored = float(c1[("출원2015-19_발명", cc)]["free5"]) * 100; stored_t = float(c1[("출원2015-19_발명", cc)]["txt5"]) * 100
    s_txt = rhu(stored_t, 1); s_free = rhu(stored, 1)
    regf = rhu(fr["reg"][cc], 1)
    has_old = ("FREE-IPC %s%%" % s_free) in ms[ln - 1]
    rec("부분 반박", "부록 %s행 C-AGREE-%s FREE-IPC 옛값 잔존" % (ln, cc), ln, "FREE-IPC %s%%·TXT %s%% 표기" % (s_free, s_txt),
        "대안 A(REG) 값 FREE %s%% (검증자 v1_out)" % regf, has_old and s_free != regf or (cc == "EP" and s_free == regf),
        "c1_label_agreement.csv(R8 저장) vs C10-T3 verify_0/v1_out.json", "EP는 FREE 53.1 동일, TXT만 변경" if cc == "EP" else "")
# TXT 대안 A 값(검증자 독립 TXT 재현율이 없으면 작업자 s2_agree를 csv 모듈로)
aw = {r["cc"]: r for r in agree_rows if r["window"] == "출원2015-19_발명"}
for cc, ln in [("US", 498), ("JP", 499), ("EP", 501)]:
    t_reg = rhu(float(aw[cc]["txt5_reg"]) * 100, 1); t_old = rhu(float(c1[("출원2015-19_발명", cc)]["txt5"]) * 100, 1)
    rec("부분 반박", "부록 %s행 TXT 옛값" % ln, ln, "TXT %s%%" % t_old, "REG TXT %s%%" % t_reg, ("TXT %s%%" % t_old) in ms[ln - 1] and t_old != t_reg, "s2_agree.csv(csv 모듈)")
rng_old = "29.6–31.1%"; rec("부분 반박", "부록 534행 C-AGREE-RANGE 옛 범위", 534, rng_old + ", 48.2–53.5%", "REG 29.4–30.7%, 48.0–53.5%",
    rng_old in ms[533] and "48.2–53.5%" in ms[533], "원고 534행 문자열")
# B04 부분(a): 상속 라벨 정의·연결 민감도
pub = [float(aw[c]["free5_pub"]) * 100 for c in ["US", "JP", "EP", "CN"]]
rec("부분 반박", "73행 '한국 패밀리 문서는 KR 라벨을 상속' — 실제 상속 라벨", 73, "KR 라벨(제공 라벨로 읽힘)",
    "KR 문서의 TXT-FREE(해설서 단독) 라벨 상속", "KR 라벨을 상속한다" in ms[72],
    r"C10\C10-T3\summary.txt 22·35행('상속 라벨(KR TXT-FREE)', 'FFAM(KR TXT-FREE 상속')", "제공 KR 라벨 상속은 진단용 PKR 팔")
rec("부분 반박", "73행 연결 민감도 누락(공개번호 연결 범위)", 73, "(원고에 없음)", "PUB %s–%s%% (US %s%%; 13자리 토큰만 쓰면 US 50.9%%)" % (rhu(min(pub), 1), rhu(max(pub), 1), rhu(pub[0], 1)),
    "공개번호" not in ms[72], "s2_agree.csv free5_pub(csv 모듈); C10-T3 verify_3 v3_out recall_free.pub", "ERRATA 원안 '47.0–53.5%'는 0.469499를 46.950으로 먼저 반올림한 이중 반올림, 원값 직접은 46.9")
# B08 부분(a): '빼도'가 동시 제거로 읽힘 — 동시 제거 변형 존재?
names = [str(v) for v in (sensj.keys() if isinstance(sensj, dict) else [r.get("variant") for r in sensj])]
txt = json.dumps(sensj, ensure_ascii=False)
both = bool(re.search(r"물가.{0,20}산업연관표|산업연관표.{0,20}물가|S4\+S5|S6", txt))
rec("부분 반박", "143행 물가·산업연관표 동시 제거 변형 존재 여부", 143, "빼도 … 0.0022 이하(동시 제거로 읽힘)", "동시 제거 변형 %s; S4/S5 각각" % ("있음" if both else "없음"),
    not both, r"C10\C10-T1\sens_results.json(작업은 .csv 사용)")
# B09 낡은 표기: 144 '따로 낸다' vs 부록 D 745 '패키지는 아직 없다'
rec("낡은 표기 반박", "144행 '패키지로 따로 낸다' ↔ 745행 '[미처리] … 패키지는 아직 없다'", 144, ms[143][:40], ms[744][:60],
    "낸다" in ms[143] and "아직 없다" in ms[744], "원고 144·745행")
# B21 부분(a): 395 T4 0.487의 지표
pj = J(L + r"\C08\C08-T3\prospective_result.json"); pjs = json.dumps(pj, ensure_ascii=False)
m = re.search(r'"skill[^"]*":\s*0\.48[67]\d*', pjs)
rec("부분 반박", "395행 T4 0.487의 지표(skill)가 빠짐", 395, "T4 4개월 전향 채점은 0.487", "원천에서 0.487 = %s" % (m.group(0)[:40] if m else "키 검색 실패"),
    ("채점은 0.487" in ms[394]) and bool(m), r"C08\C08-T3\prospective_result.json", "AUC는 0.798")
# B23 부분(a): 410 N6 +0.006–0.010의 원인(동명 복원) 누락
f177 = [l for l in open(L + r"\FINDINGS.txt", encoding="utf-8") if l.startswith("[C07][C07-T2][warning] (verify_0 확인) 정의 불일치 3건")]
rec("부분 반박", "410행 'N6(미등록 포함)' — '동명 복원' 삭제로 E-C07T2-03 오귀속 재발", 410, "N6(미등록 포함) +0.006–0.010",
    "N6 +0.006–0.010은 동명 상호 배제분 복원(C06-T3), 등록 공란만 복원은 +0.003", "N6(미등록 포함)" in ms[409] and "동명" not in ms[409] and bool(f177),
    r"FINDINGS C07-T2 정의 불일치 (2); ERRATA E-C07T2-03; drafts\manuscript_for_v11.md 410행")
# B32/B33/C04 낡은 표기: 출처 열
for ln, want in [(519, "C10-T3"), (637, "C10-T3"), (482, "C11-T2")]:
    cells = [c.strip() for c in ms[ln - 1].strip().strip("|").split("|")]
    rec("낡은 표기 반박", "%d행 출처 열(값은 %s 재계산)" % (ln, want), ln, cells[-1][-60:], "값의 실제 출처 %s" % want, want not in ms[ln - 1], "원고 %d행 문자열" % ln)
# B40 낡은 표기: 743 제목 '셋' vs 244 '다섯'
rec("낡은 표기 반박", "743행 제목 '공개 대안 셋' ↔ 244행 '다섯'", 743, "셋", "다섯", "대안 셋" in ms[742] and "대안 다섯" in ms[243], "원고 244·743행")
# 그림 1 낡은 표기: PNG 문구(ECOS) — make_diagrams.py·fig01 CSV(문구 원천)와 PNG 픽셀 차 영역
md = open(AN + r"\보고서_20261012\figs\make_diagrams.py", encoding="utf-8").read()
fcsv = open(AN + r"\보고서_20261012\figs\fig01_framework.csv", encoding="utf-8-sig").read()
rec("낡은 표기 반박", "그림 1 PNG '산업연관표·물가 (ECOS)' ↔ 표 3·표 13 '한국은행 접속 산업연관표'", 81, "산업연관표·물가 (ECOS)", "표 3(90행) 한국은행 접속 산업연관표",
    "산업연관표·물가 (ECOS)" in md and "산업연관표·물가 (ECOS)" in fcsv and "한국은행 접속 산업연관표" in ms[89], "figs\\make_diagrams.py 55행, fig01_framework.csv; build_test\\figs_fix 재렌더 차 영역 (288,270)-(609,302)")
# facts 옛값(부분 c-facts)
for key, old in [("C-AGREE-US", "51.0%"), ("C-AGREE-JP", "48.2%"), ("C-AGREE-RANGE", "29.6–31.1%"), ("C-RCA-FREE-IPC", "0.571"), ("C-T3-RET", "25%"),
                 ("C-T3-drop_ros-hyb_txt", "+0.0239"), ("T3NC-prov_vs_txt", "+0.0020"), ("T3NC-MG", "0.7947"), ("MAP-KR-RTA", "0.20–0.36"), ("B-P50-FREE50", "0.002~0.004"), ("C-LAND-free_F", "0.360")]:
    rs = facts.get(key, [])
    hit = [fn for fn, r in rs if old in json.dumps(r, ensure_ascii=False)]
    sup = [fn for fn, r in rs if fn == "facts_supplement.csv"]
    rec("부분(c) 반박", "facts %s 옛값 %s" % (key, old), "", old, "facts 행 %d개, 옛값 포함 %s, supplement 정정 행 %s" % (len(rs), hit, "있음" if sup else "없음"),
        bool(hit) and not sup, "facts\\facts.csv·facts_supplement.csv(csv 모듈)")
# 부록 A-COVER ESS 318 (저장 318.5)
s3m = open(L + r"\C06\C06-T1\out\s3_meta.json", encoding="utf-8").read()
rec("부분 반박", "부록 625행 A-COVER 'ESS 318'", 625, "ESS 318", "저장 ESS 318.5(1자리) → 사사오입 319, 은행가 반올림 318", '"ESS": 318.5' in s3m and "ESS 318" in ms[624],
    r"C06\C06-T1\out\s3_meta.json", "원값 1자리 저장이라 판정은 '이중 반올림 가능'")
# ---------- (2) 정확 하위 블록 무작위 15개 독립 재계산 ----------
def mean(a): return sum(a) / len(a)
ao = v4b["auc_by_origin"]; am = v4b["auc_mean"]
T = {}
T["B01/B10/B22 AUC 0.002~0.005"] = lambda: (rhu(min(mean([v1p["auc"][o]["P50"] - v1p["auc"][o][k] for o in v1p["auc"]]) for k in ["FREE50", "TXT50", "FREEIPC50", "HSC1_50"]), 3) + "~" +
                                            rhu(max(mean([v1p["auc"][o]["P50"] - v1p["auc"][o][k] for o in v1p["auc"]]) for k in ["FREE50", "TXT50", "FREEIPC50", "HSC1_50"]), 3), "0.002~0.005", 13)
T["B13 FREE-IPC +0.0031"] = lambda: ("+" + rhu(mean([v1p["auc"][o]["P50"] - v1p["auc"][o]["FREEIPC50"] for o in v1p["auc"]]), 4), "+0.0031", 336)
T["B13 H-sc-1 +0.0045"] = lambda: ("+" + rhu(mean([v1p["auc"][o]["P50"] - v1p["auc"][o]["HSC1_50"] for o in v1p["auc"]]), 4), "+0.0045", 336)
T["B13 5/5"] = lambda: ("%d/5" % min(sum(v1p["auc"][o]["P50"] > v1p["auc"][o][k] for o in v1p["auc"]) for k in ["FREEIPC50", "HSC1_50"]), "5/5", 336)
T["B02 US 50.5%"] = lambda: (rhu(fr["reg"]["US"], 1) + "%", "50.5%", 68)
T["B03 JP 48.0%"] = lambda: (rhu(fr["reg"]["JP"], 1) + "%", "48.0%", 69)
T["B05 48.0–53.5%"] = lambda: (rhu(min(fr["reg"].values()), 1) + "–" + rhu(max(fr["reg"].values()), 1) + "%", "48.0–53.5%", 75)
T["B38 상속 없음 52.3/49.4/55.0"] = lambda: ("US %s%%, JP %s%%, EP %s%%" % (rhu(fr["pure"]["US"], 1), rhu(fr["pure"]["JP"], 1), rhu(fr["pure"]["EP"], 1)), "US 52.3%, JP 49.4%, EP 55.0%", 648)
T["B15 보존율 23–29%"] = lambda: (rhu((am["free_reg"] - am["kr_rebuild"]) / (am["drop_ros"] - am["kr_rebuild"]) * 100, 0) + "–" + rhu((am["txt_reg"] - am["kr_rebuild"]) / (am["drop_ros"] - am["kr_rebuild"]) * 100, 0) + "%", "23–29%", 341)
T["B15 +0.0242~+0.0262"] = lambda: ("+" + rhu(mean([a - b for a, b in zip(ao["drop_ros"], ao["txt_reg"])]), 4) + "~+" + rhu(mean([a - b for a, b in zip(ao["drop_ros"], ao["free_reg"])]), 4), "+0.0242~+0.0262", 341)
T["B15 9/9"] = lambda: ("%d/9" % min(sum(a > b for a, b in zip(ao["drop_ros"], ao[k])) for k in ["txt_reg", "free_reg"]), "9/9", 341)
T["B32 hyb_txt 0.7626"] = lambda: (rhu(am["txt_reg"], 4), "0.7626", 519)
T["B32 hyb_free 0.7605"] = lambda: (rhu(am["free_reg"], 4), "0.7605", 519)
T["B33 상실 71–77%"] = lambda: (rhu(100 - (am["txt_reg"] - am["kr_rebuild"]) / (am["drop_ros"] - am["kr_rebuild"]) * 100, 0) + "–" + rhu(100 - (am["free_reg"] - am["kr_rebuild"]) / (am["drop_ros"] - am["kr_rebuild"]) * 100, 0) + "%", "71–77%", 637)
T["B16 공개 0.444~0.608(FREE)"] = lambda: (rhu(v1["rho_free"]["reg"], 3), "0.608", 342)
T["B37 공개번호 0.554·상속 없음 0.596"] = lambda: ("상속 없음 " + rhu(v1["rho_free"]["pure"], 3), "상속 없음 0.596", 642)
T["C01 ρ 0.19–0.37"] = lambda: (rhu(c11v1["range_168"]["REG_raw"]["min_raw"], 2) + "–" + rhu(c11v1["range_168"]["REG_raw"]["max_raw"], 2), "0.19–0.37", 168)
T["C02 +0.0017 [−0.0004, +0.0038], 6/9"] = lambda: ("+%s [−%s, +%s], %d/9" % (rhu(mean(c11r2["reg_mine"]["same_seed"]["by_origin"]), 4), rhu(-c11r2["reg_mine"]["same_seed"]["lo"], 4), rhu(c11r2["reg_mine"]["same_seed"]["hi"], 4), sum(x > 0 for x in c11r2["reg_mine"]["same_seed"]["by_origin"])), "+0.0017 [−0.0004, +0.0038], 6/9", 355)
T["C07 TXT 0.7950"] = lambda: ("TXT " + rhu(c11auc["reg"]["mean_auc_sk"], 4), "TXT 0.7950", 573)
T["C04 free_F 31/16/5/0.373"] = lambda: (next("RTA>1 %s개, 한국 1위 %s개, 제공 1위 7개 중 유지 %s, 한국 RTA ρ(제공 대비) %s" % (r["kr_rta_gt1"], r["kr_leader"], r["kept"], rhu(float(r["rho_raw"]), 3)) for r in v3rows if r["arm"] in ("reg", "REG") and r["item"] == "free_F"), "RTA>1 31개, 한국 1위 16개, 제공 1위 7개 중 유지 5, 한국 RTA ρ(제공 대비) 0.373", 482)
T["C06 txt_F 29/12/3/0.195"] = lambda: (next("RTA>1 %s개, 한국 1위 %s개, 제공 1위 7개 중 유지 %s, 한국 RTA ρ(제공 대비) %s" % (r["kr_rta_gt1"], r["kr_leader"], r["kept"], rhu(float(r["rho_raw"]), 3)) for r in v3rows if r["arm"] in ("reg", "REG") and r["item"] == "txt_F"), "RTA>1 29개, 한국 1위 12개, 제공 1위 7개 중 유지 3, 한국 RTA ρ(제공 대비) 0.195", 547)
T["B08 0.0022"] = lambda: (rhu(max(abs(sensj["variants"][k]["d_skill"]) for k in ["S4", "S5"]), 4), "0.0022", 143)
FIRM = L + r"\C10\C10-T2\verify_1\v1_boot_firm.json"
T["B12 P50−FREE50 법인 군집 CI(검증자 B=300)"] = lambda: ("[+%s, +%s]" % (rhu(J(FIRM)["res"]["P50-FREE50"]["ci95"][0], 4), rhu(J(FIRM)["res"]["P50-FREE50"]["ci95"][1], 4)), "[+0.0004, +0.0042]", 334)
T["B31 165"] = lambda: (str(J(L + r"\C10\C10-T1\verify_3\v3_io_check_out.json")["n_3digit_cols_2010"]), "165", 507)
random.seed(20261013)
keys = sorted(T.keys()); pick = sorted(random.sample(keys, 15))
for k in keys:
    try:
        got, exp, ln = T[k]()
        ok = exp in ms[ln - 1] and (got == exp or got in exp or exp.endswith(got))
        rec("정확 재계산" + ("(무작위 15)" if k in pick else "(추가)"), k, ln, exp, got, ok, "독립 경로(verify_*·원점별 AUC·csv 모듈)")
    except Exception as e:
        rec("정확 재계산" + ("(무작위 15)" if k in pick else "(추가)"), k, "", "", "실패: %r" % e, False, "")
with open(W + r"\verify_0\v0_results.csv", "w", encoding="utf-8-sig", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
for o in out: print(o["kind"], "|", o["item"], "|", o["manuscript"][:50] if isinstance(o["manuscript"], str) else o["manuscript"], "|", str(o["independent"])[:80], "|", o["agree"])
print("pick:", pick)

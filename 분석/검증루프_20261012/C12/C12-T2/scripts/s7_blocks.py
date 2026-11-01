# C12-T2 s7: 변경 블록(v11→v12 41행, v12→v13 8행)의 하위 블록 숫자를 근거 원천에서 다시 계산(작업 경로: pandas/json)
# 표기 자릿수로 '한 번' 반올림(사사오입)한 문자열이 기준 원고(v13) 해당 줄에 있는지 확인. 출력: cache\s7_subblocks.csv
import sys, json, re
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = AN + r"\검증루프_20261012"
MC = AN + r"\모델경쟁_20260927"
W = L + r"\C12\C12-T2"
ms = open(AN + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().split("\n")
def R(x, d):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
def f(x, d, sign=False, pct=False):
    v = float(x) * (100 if pct else 1)
    s = str(R(abs(v), d))
    if sign: s = ("+" if v >= 0 else "−") + s
    elif v < 0: s = "−" + s
    return s
J = lambda p: json.load(open(p, encoding="utf-8"))
agree = pd.read_csv(L + r"\C10\C10-T3\res\s2_agree.csv", encoding="utf-8-sig")
aw = agree[agree.window == "출원2015-19_발명"].set_index("cc")
rho = pd.read_csv(L + r"\C10\C10-T3\res\s3_rho_table.csv", encoding="utf-8-sig").set_index(["arm", "lab"])
t3 = J(L + r"\C10\C10-T3\res\s6_t3_summary.json")
boot = pd.read_csv(L + r"\C10\C10-T2\res\boot_summary.csv", encoding="utf-8-sig").set_index("pair")
cpool = pd.read_csv(MC + r"\T1_산업대기실\TXT_R9\res\C_pool_summary.csv", encoding="utf-8-sig")
sens = pd.read_csv(L + r"\C10\C10-T1\sens_results.csv", encoding="utf-8-sig").set_index("variant")
io = J(L + r"\C10\C10-T1\verify_3\v3_io_check_out.json")
land = pd.read_csv(L + r"\C11\C11-T2\res\s7_landscape.csv", encoding="utf-8-sig").set_index("name")
mg = {a: J(L + r"\C11\C11-T2\res\s5_summary_mg__%s.json" % a) for a in ["reg", "pub", "pure"]}
mgm = {m["model"]: m for m in mg["reg"]["models"]}
F = [c for c in ["US", "JP", "CN", "EP"]]
fr = aw.loc[F, "free5_reg"]; tr = aw.loc[F, "txt5_reg"]; fp = aw.loc[F, "free5_pure"]; fpub = aw.loc[F, "free5_pub"]
def rng(a, b, d, pct=False, sep="–"):
    return f(a, d, pct=pct) + sep + f(b, d, pct=pct)
# B 근거 쌍(공개 대안): 연계표 CW27(C_pool), TXT, FREE, FREE-IPC, H-sc-1
cw = cpool[cpool.astype(str).apply(lambda r: r.str.contains("CW27|cw27|CW", regex=True)).any(axis=1)]
print(cpool.head(12).to_string()[:1500])
S = []
def add(bid, line, label, val, evidence):
    present = val in ms[line - 1]
    S.append(dict(block=bid, line=line, sub=label, recomputed=val, in_manuscript=present, evidence=evidence))
Bpairs = {"FREE50": boot.loc["P50-FREE50", "mean_d"], "TXT50": boot.loc["P50-TXT50", "mean_d"], "FREEIPC": boot.loc["P50-FREEIPC50", "mean_d"], "HSC1": boot.loc["P50-HSC1_50", "mean_d"]}
add("B01", 13, "AUC 0.002~0.005(근거 B 공개 대안 최소~최대)", rng(min(Bpairs.values()), max(Bpairs.values()), 3, sep="~"), "C10-T2 res/boot_summary.csv mean_d; 연계표 대비 +0.0041(C_pool)도 범위 안")
add("B10", 244, "AUC 0.002~0.005", rng(min(Bpairs.values()), max(Bpairs.values()), 3, sep="~"), "같음")
add("B22", 407, "AUC 0.002~0.005", rng(min(Bpairs.values()), max(Bpairs.values()), 3, sep="~"), "같음")
add("B02", 68, "US FREE-IPC", f(fr["US"], 1, pct=True) + "%", "C10-T3 res/s2_agree.csv free5_reg")
add("B03", 69, "JP FREE-IPC", f(fr["JP"], 1, pct=True) + "%", "같음")
add("B05", 75, "해외 FREE-IPC 범위", rng(fr.min(), fr.max(), 1, pct=True) + "%", "같음")
add("B14", 340, "텍스트+IPC 표 범위", rng(tr.min(), tr.max(), 1, pct=True) + "%", "s2_agree txt5_reg")
add("B14", 340, "해설서 단독+IPC 표 범위", rng(fr.min(), fr.max(), 1, pct=True) + "%", "s2_agree free5_reg")
add("B14", 340, "JP만 50% 미만", "JP만 50% 미만" if (fr < 0.5).sum() == 1 and fr.idxmin() == "JP" else "깨짐", "s2_agree free5_reg")
add("B39", 677, "해외 48.0–53.5%", rng(fr.min(), fr.max(), 1, pct=True) + "%", "s2_agree")
add("B38", 648, "상속 없음 US", f(fp["US"], 1, pct=True) + "%", "s2_agree free5_pure")
add("B38", 648, "상속 없음 JP", f(fp["JP"], 1, pct=True) + "%", "s2_agree free5_pure")
add("B38", 648, "상속 없음 EP", f(fp["EP"], 1, pct=True) + "%", "s2_agree free5_pure")
add("B04", 73, "(참고, 원고에 없음) 공개번호 연결 범위", rng(fpub.min(), fpub.max(), 1, pct=True) + "%", "s2_agree free5_pub")
s4 = abs(sens.loc["S4", "d_skill"]); s5 = abs(sens.loc["S5", "d_skill"])
add("B08", 143, "물가·IO 제거 skill 변화 상한", f(max(s4, s5), 4), "C10-T1 sens_results.csv S4 %.5f, S5 %.5f (각각 제거; 동시 제거 변형 없음)" % (-s4, -s5))
fb = boot.loc["P50-FREE50"]
add("B12", 334, "이원 군집 CI 0 포함", "0 포함" if fb.firm_sc_lo < 0 < fb.firm_sc_hi else "0 미포함", "boot_summary firm_sc [%.5f, %.5f]" % (fb.firm_sc_lo, fb.firm_sc_hi))
add("B13", 336, "FREE-IPC 대비", f(Bpairs["FREEIPC"], 4, sign=True), "boot_summary P50-FREEIPC50")
add("B13", 336, "H-sc-1 대비", f(Bpairs["HSC1"], 4, sign=True), "boot_summary P50-HSC1_50")
add("B13", 336, "5/5", "5/5" if boot.loc["P50-FREEIPC50", "wins"] == 5 and boot.loc["P50-HSC1_50", "wins"] == 5 else "x", "boot_summary wins")
add("B13", 336, "이원 군집 CI 하한>0", "CI 하한>0" if min(boot.loc["P50-FREEIPC50", "firm_sc_lo"], boot.loc["P50-HSC1_50", "firm_sc_lo"]) > 0 else "x", "boot_summary firm_sc_lo")
add("B17", 354, "이원 군집 CI 0 포함", "0 포함" if fb.firm_sc_lo < 0 else "x", "boot_summary")
reg = t3["reg"]
add("B15", 341, "보존율 범위", f(reg["retention"]["free"], 0, pct=True) + "–" + f(reg["retention"]["txt"], 0, pct=True) + "%", "C10-T3 res/s6_t3_summary.json reg.retention")
add("B15", 341, "제공 − 공개 범위", f(reg["drop_ros−hyb_txt_reg"]["dAUC_raw"], 4, sign=True) + "~" + f(reg["drop_ros−hyb_free_reg"]["dAUC_raw"], 4, sign=True), "s6 reg dAUC_raw")
add("B15", 341, "해외 몫 AUC", f(reg["drop_ros−kr_rebuild"]["dAUC_raw"], 4, sign=True), "s6 drop_ros−kr_rebuild")
add("B19", 358, "TXT / FREE-IPC", f(reg["drop_ros−hyb_txt_reg"]["dAUC_raw"], 4, sign=True) + " / " + f(reg["drop_ros−hyb_free_reg"]["dAUC_raw"], 4, sign=True), "s6 reg")
add("B32", 519, "hyb_txt", f(t3["auc"]["hyb_txt_reg"], 4), "s6 auc")
add("B32", 519, "hyb_free", f(t3["auc"]["hyb_free_reg"], 4), "s6 auc")
add("B33", 637, "hyb_free 보존", "hyb_free " + f(reg["retention"]["free"], 0, pct=True) + "%", "s6 retention")
add("B33", 637, "상실 범위", f(1 - reg["retention"]["txt"], 0, pct=True) + "–" + f(1 - reg["retention"]["free"], 0, pct=True) + "%", "1 − retention")
add("B34", 638, "CI", "[" + f(reg["drop_ros−hyb_txt_reg"]["ci"][0], 4, sign=True) + ", " + f(reg["drop_ros−hyb_txt_reg"]["ci"][1], 4, sign=True) + "]", "s6 ci(저장 4자리)")
add("B35", 639, "CI", "[" + f(reg["drop_ros−hyb_free_reg"]["ci"][0], 4, sign=True) + ", " + f(reg["drop_ros−hyb_free_reg"]["ci"][1], 4, sign=True) + "]", "s6 ci")
rf = rho.loc[("reg", "free")]; rt = rho.loc[("reg", "txt")]
add("B16", 342, "공개 범위", f(rt.rho_alt_raw, 3) + "~" + f(rf.rho_alt_raw, 3), "C10-T3 res/s3_rho_table.csv reg")
cd = json.loads(rf.ci_d.replace("'", '"'))
add("B18", 357, "Δ [CI]", f(rf.d, 3, sign=True).replace("+", "") if rf.d < 0 else f(rf.d, 3, sign=True), "s3_rho_table reg free d")
add("B18", 357, "CI", "[" + f(cd[0], 3, sign=True) + ", " + f(cd[1], 3, sign=True) + "]", "s3_rho_table ci_d(저장 3자리)")
ca = json.loads(rt.ci_alt); cdt = json.loads(rt.ci_d)
add("B36", 641, "TXT ρ [CI]", f(rt.rho_alt_raw, 3) + " [" + f(ca[0], 3) + ", " + f(ca[1], 3) + "]", "s3_rho_table reg txt")
add("B36", 641, "Δ [CI]", f(rt.d, 3, sign=True) + " [" + f(cdt[0], 3, sign=True) + ", " + f(cdt[1], 3, sign=True) + "]", "s3_rho_table")
caf = json.loads(rf.ci_alt)
add("B37", 642, "FREE ρ [CI]", f(rf.rho_alt_raw, 3) + " [" + f(caf[0], 3) + ", " + f(caf[1], 3) + "]", "s3_rho_table reg free")
add("B37", 642, "공개번호 연결", "공개번호 연결 " + f(rho.loc[("pub", "free")].rho_alt_raw, 3), "s3_rho_table pub free")
add("B37", 642, "상속 없음", "상속 없음 " + f(rho.loc[("pure", "free")].rho_alt_raw, 3), "s3_rho_table pure free")
add("B31", 507, "통합소분류 165", "통합소분류 165" if io["n_3digit_cols_2010"] == 165 else "x", "C10-T1 verify_3/v3_io_check_out.json")
# v12→v13
Fr = [n for n in land.index if n.endswith("_F_reg")] + ["cweq_F"]
Fr2 = [n for n in land.index if n.endswith("_F_reg")]
add("C01", 168, "ρ 범위(전 라벨 공개, F)", f(land.loc[Fr, "rho_KR_raw"].min(), 2) + "–" + f(land.loc[Fr, "rho_KR_raw"].max(), 2), "C11-T2 res/s7_landscape.csv F_reg+cweq_F")
add("C01", 168, "한국 1위 범위", "%d–%d개" % (land.loc[Fr, "kr_leader"].min(), land.loc[Fr, "kr_leader"].max()), "s7_landscape.csv kr_leader")
p = mg["reg"]["pairs"]["prov_vs_txt"]
add("C02", 355, "Δ [CI], wins", f(p["dAUC"], 4, sign=True) + " [" + f(p["ci"][0], 4, sign=True) + ", " + f(p["ci"][1], 4, sign=True) + "], %d/%d" % (p["wins"], p["n_orig"]), "C11-T2 res/s5_summary_mg__reg.json")
add("C08", 636, "Δ [CI], wins", f(p["dAUC"], 4, sign=True) + " [" + f(p["ci"][0], 4, sign=True) + ", " + f(p["ci"][1], 4, sign=True) + "], %d/%d" % (p["wins"], p["n_orig"]), "같음")
add("C08", 636, "연결 정의 범위", f(mg["pub"]["pairs"]["prov_vs_txt"]["dAUC"], 4, sign=True) + "~" + f(mg["pure"]["pairs"]["prov_vs_txt"]["dAUC"], 4, sign=True), "s5_summary_mg__pub/pure.json")
add("C08", 636, "상속 없음 CI 하한>0", "CI 하한>0" if mg["pure"]["pairs"]["prov_vs_txt"]["ci"][0] > 0 else "x", "s5_summary_mg__pure.json")
add("C07", 573, "TXT", "TXT " + f(mgm["txt"]["auc"], 4), "s5_summary_mg__reg.json models.txt")
add("C04", 482, "free_F", "RTA>1 %d개, 한국 1위 %d개, 제공 1위 7개 중 유지 %d, 한국 RTA ρ(제공 대비) %s" % (land.loc["free_F_reg", "kr_rta_gt1"], land.loc["free_F_reg", "kr_leader"], land.loc["free_F_reg", "kept"], f(land.loc["free_F_reg", "rho_KR_raw"], 3)), "s7_landscape.csv free_F_reg")
add("C06", 547, "txt_F", "RTA>1 %d개, 한국 1위 %d개, 제공 1위 7개 중 유지 %d, 한국 RTA ρ(제공 대비) %s" % (land.loc["txt_F_reg", "kr_rta_gt1"], land.loc["txt_F_reg", "kr_leader"], land.loc["txt_F_reg", "kept"], f(land.loc["txt_F_reg", "rho_KR_raw"], 3)), "s7_landscape.csv txt_F_reg")
df = pd.DataFrame(S)
df.to_csv(W + r"\cache\s7_subblocks.csv", index=False, encoding="utf-8-sig")
print(df.to_string())
print("not in manuscript:", (~df.in_manuscript).sum())

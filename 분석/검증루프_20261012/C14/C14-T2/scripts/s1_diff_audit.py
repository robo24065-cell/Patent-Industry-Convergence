# C14-T2 s1: v14↔v15 변경 11행(diff 출력 21줄) 감사 — ERRATA 원안 문구 대조 + 원천 결과 파일에서 숫자 재계산 → diff_audit.csv, cache/s1_recompute.json
import sys, re, json, difflib, csv
import pandas as pd, numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = B + r"\검증루프_20261012"
W = L + r"\C14\C14-T2"
v14 = open(B + r"\보고서_20261012\drafts\manuscript_for_v14.md", encoding="utf-8").read().split("\n")
v15 = open(B + r"\보고서_20261012\manuscript_v3.md", encoding="utf-8").read().split("\n")
er = open(L + r"\ERRATA.txt", encoding="utf-8").read().split("\n")
sm = difflib.SequenceMatcher(None, v14, v15, autojunk=False)
pairs = []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == "equal": continue
    for k in range(max(i2 - i1, j2 - j1)):
        pairs.append((op, i1 + k + 1 if i1 + k < i2 else None, j1 + k + 1 if j1 + k < j2 else None))
n_diff_lines = sum((i2 - i1) + (j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
# ---------- 원천 재계산 ----------
rc = {}
bs = pd.read_csv(L + r"\C10\C10-T2\res\boot_summary.csv", encoding="utf-8-sig").set_index("pair")
five = ["P50-FREE50", "P50-TXT50", "P50-FREEIPC50", "P50-HSC1_50", "P50-HMG3_50"]
rc["L13_auc_range"] = [round(bs.loc[p, "mean_d"], 5) for p in five]
rc["L13_free50_firm_sc_ci"] = [round(bs.loc["P50-FREE50", "firm_sc_lo"], 5), round(bs.loc["P50-FREE50", "firm_sc_hi"], 5)]
rc["L13_freeipc_hsc1_all6_lo_gt0"] = bool(all(bs.loc[p, c] > 0 for p in ["P50-FREEIPC50", "P50-HSC1_50"] for c in ["firm_lo", "ksic_t_lo", "ksic_f_lo", "sc_lo", "firm_sc_lo", "firm_ksic_t_lo"]))
v1 = pd.read_csv(L + r"\C12\C12-T3\verify_1\v1_summary.csv", encoding="utf-8-sig")
f = v1[v1.part == "final2021"].set_index("exp")
rc["L219_final_no_ext"] = dict(d_mean=round(f.loc["no_ext", "d_mean"], 4), d_pos=int(f.loc["no_ext", "d_pos"]), d_min=round(f.loc["no_ext", "d_min"], 4))
rc["L219_final_no_pi"] = dict(d_mean=round(f.loc["no_pi", "d_mean"], 4), d_pos=int(f.loc["no_pi", "d_pos"]))
rc["L219_final_no_ros"] = dict(d_mean=round(f.loc["no_ros", "d_mean"], 4), d_pos=int(f.loc["no_ros", "d_pos"]))
dv = v1[v1.part == "dev2010_18"].set_index("exp")
rc["L219_dev"] = dict(no_pi=round(dv.loc["no_pi", "d_mean"], 4), no_ros=round(dv.loc["no_ros", "d_mean"], 4))
c10t1 = pd.read_csv(L + r"\C10\C10-T1\sens_results.csv", encoding="utf-8-sig").set_index("variant")
rc["L143_T1_S4_S5"] = [round(c10t1.loc["S4", "d_skill"], 5), round(c10t1.loc["S5", "d_skill"], 5)]
rc["L143_T3_pi_vintage_S2"] = round(pd.read_csv(L + r"\C12\C12-T3\b_sens.csv", encoding="utf-8-sig").query("scenario=='S2' and seed=='mean(0-4)'")["d_skill_vs_S0"].iloc[0], 4)
v5 = json.load(open(L + r"\C11\C11-T2\verify_5\v5_out.json", encoding="utf-8"))
rc["L168_base"] = [round(v5["range_168"]["REG_base"]["min_raw"], 4), round(v5["range_168"]["REG_base"]["max_raw"], 4)]
rc["L168_tie_full_max"] = round(v5["range_168"]["REG_tie_desc_full"]["max_raw"], 4)
rs = pd.read_csv(L + r"\C01\C01-T3\rescoring_sensitivity.csv", encoding="utf-8-sig")
li = rs[rs.label == "L_inv"].set_index("metric")
rc["L286"] = dict(cap_champ=li.loc["cap_champ", "point"], cap_SB=li.loc["cap_SB", "point"], dcap_SB=li.loc["dcap_SB", "point"],
                  dcap_SB_lo_owner=li.loc["dcap_SB", "ci_owner_lo"], dcap_SB_lo_bundle=li.loc["dcap_SB", "ci_bundle_lo"], dcap_SB_hi_owner=li.loc["dcap_SB", "ci_owner_hi"])
mg = json.load(open(L + r"\C11\C11-T2\res\s5_summary_mg__reg.json", encoding="utf-8"))
rc["L336_table10_row"] = [l for l in v15 if l.startswith("| T3 메인그룹 성장")][0][:120]
s8 = pd.read_csv(L + r"\C08\C08-T1\sensitivity.csv", encoding="utf-8-sig")
g8 = s8[s8.item.str.contains("연평균 로그 성장")]; o = g8[g8.item.str.contains("OLED")].iloc[0]; lc = g8[g8.item.str.contains("액정")].iloc[0]
rc["L374"] = dict(LCD_S2=lc["S2_aux"], OLED_S2=o["S2_aux"], OLED_note=o["note"])
nd = json.load(open(L + r"\C07\C07-T3\verify_0\v02_out.json", encoding="utf-8"))
rc["L387"] = dict(champ_new=nd["champ"]["new"], M0_new=nd["M0"]["new"], SB_new=nd["SB"]["new"], pos_new=nd["pos_new"])
a = pd.read_csv(L + r"\C07\C07-T3\ksic_rank_check.csv", encoding="utf-8-sig"); b = pd.read_csv(L + r"\C08\C08-T3\industry_rank_check.csv", encoding="utf-8-sig")
rc["L395"] = dict(N8_index=round(spearmanr(a.rank_index_meanp, a.rank_realized_rate)[0], 3), N8_persist=round(spearmanr(a.rank_persist, a.rank_realized_rate)[0], 3),
                  T4_index=round(spearmanr(b.rank_index, b.rank_realized)[0], 3), T4_persist=round(spearmanr(b.rank_persist, b.rank_realized)[0], 3),
                  N8_count_index_vs_count=round(spearmanr(a.rank_index_exp, a.rank_realized_count)[0], 3), N8_count_persist=round(spearmanr(a.rank_persist, a.rank_realized_count)[0], 3))
sb = pd.read_csv(L + r"\C13\C13-T2\seed_bundles.csv", encoding="utf-8-sig"); vd = pd.read_csv(L + r"\C13\C13-T2\variance_decomp.csv", encoding="utf-8-sig")
rc["L534"] = dict(bundles=sb.groupby("track").size().to_dict(), n_ci_lo_gt0=int((sb.ci_lo > 0).sum()), n_total=int(len(sb)),
                  strong_class=sb.groupby("track").strong_class.unique().apply(list).to_dict(),
                  T4_strong_lo=[round(sb[sb.track == "T4"].strong_ci_lo.min(), 4), round(sb[sb.track == "T4"].strong_ci_lo.max(), 4)],
                  ratio=[round(vd.ratio_seed_sd_to_boot_se.min(), 4), round(vd.ratio_seed_sd_to_boot_se.max(), 4)],
                  N8N9_seeds=sb[sb.track.isin(["N8", "N9"])].seeds.unique().tolist()[:3])
json.dump(rc, open(W + r"\cache\s1_recompute.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in rc.items(): print(k, v)
# ---------- 행별 판정 ----------
def E(n): return er[n - 1]
A = {  # v15 줄: (ERRATA 줄들, 원안 핵심 새 문구, 재계산 근거 키, 판정, 근거 파일, 설명)
 13: ([841], ["AUC 0.002~0.005로 작다(해설서 단독 대비는 이원 군집 CI가 0 포함)."], "L13_*", "정확",
      r"C10\C10-T2\res\boot_summary.csv", "ΔAUC 다섯 쌍 0.0024–0.0045(→0.002~0.005), P50−FREE50 법인×서브클래스 [−0.0006, +0.0057] 0 포함, FREE-IPC·H-sc-1은 6가지 군집 모두 하한>0. 원안과 문자 동일."),
 79: ([842, 838], ["① 모든 트랙의 모집단·단위는 제공 데이터에서 정의한다(N4 폐업·N7 과제 진입 결과는 외부 사건)", "(예외: 물가·산업연관표·생산지수의 현행 빈티지, 2.5)"], "—", "부분(과장 잔존)+서식 경미",
      r"원고 99·96·255–267행; C13\C13-T3\claim_map.csv K020",
      "원안과 문자 동일하나 (a) 원안 괄호 뒤 기존 괄호가 붙어 ')(' 연속(서식) (b) N6(NTIS 수행기업)·N9(NTIS 과제)의 모집단·단위는 NTIS에서 오고(255–267행), N5 결과(ETSI 선언, 99행)도 외부 사건인데 괄호는 N4·N7만 든다 → '모든 트랙의 모집단·단위는 제공 데이터에서 정의' 문장이 여전히 2개 트랙에서 틀림. 생산지수 예외(E-C13T3-04)는 정확."),
 143: ([838, 803], ["(물가·산업연관표·생산지수는 원점 이후 공표된 현행 빈티지"], "L143_*", "부분(의미 변화)",
      r"C10\C10-T1\sens_results.csv S4·S5; C12\C12-T3\b_sens.csv S2, verify_1\v1_summary.csv",
      "원안(E-C12T3-04 짧은 판)과 문자 동일. 그러나 같은 괄호의 '각각 빼도 T1 skill 변화 0.0022 이하'가 생산지수까지 덮게 됨: 생산지수는 T1 피처가 아니고(T1 입력 229행: 디자인·실용신안·물가·산업연관표) T3 피처이며, T3에서 빼면 최종 +0.0082(개발 −0.0010)로 '변화 0.0022 이하'가 아님. 숫자 0.0022(S5 −0.00221)는 정확."),
 168: ([832], ["주) RTA는", "분야 수록 범위를 재가중하거나(1위 9개, RTA>1 27–29개) 산업별 1위 출원인을 빼도", "0.19–0.37(동률 규칙에 따라 최대 0.48)이며 어느 쪽이 맞는지 판정하지 못했다."], "L168_*", "정확",
      r"C11\C11-T2\verify_5\v5_out.json, v5_tiecheck.json", "기준 0.1945–0.3734(→0.19–0.37), 동률 전면 반전 최대 0.4755(→0.48). (a)+(b) 원안과 문자 동일. '1위 9개'에서 '한국'이 빠졌으나 문맥상 한국(원안 그대로)."),
 219: ([831, 802], ["해외 신호가 그 위에 더한다(외부 데이터 몫은 작아 최종시험에서는 빼도 같거나 오른다)."], "L219_*", "정확",
      r"C12\C12-T3\verify_1\v1_summary.csv, b_sens.csv", "최종 외부 전체 제거 +0.0081(5/5 seed 양수, 최소 +0.0051), 생산지수 +0.0082, RoS +0.0041 → '같거나 오른다' 성립. 개발 원점에서는 −0.0010·−0.0039(문장은 '최종시험에서는'으로 한정). 원안과 문자 동일."),
 286: ([839], ["(권리자 청구 추정 약 9%). 무효심판만으로 재채점해도 상위 1% 포착은 39.3%(과거 분쟁 기준선 33.3%, 차이 하한 0)다."], "L286", "정확",
      r"C01\C01-T3\rescoring_sensitivity.csv L_inv", "포착 0.3929·SB 0.3333, 차이 +0.0595 [0.0000, 0.1471](권리자)·[0.0000, 0.1515](묶음). 압축판 원안과 문자 동일."),
 336: ([833], ["T3 메인그룹 성장(경계)과 N9·N6에서는 같은 우위가 없었다."], "L336_table10_row", "정확",
      r"원고 355행 표 10; C11\C11-T2\res\s5_summary_mg__reg.json", "표 10 '같다(경계)' +0.0017 [−0.0004, +0.0038] 6/9와 일치. 원안과 문자 동일."),
 374: ([837], ["(연계표 대비 기술통계; OLED 증가는 두 기업 출원에 기대어 한 기업 몫을 20%로 제한하면 사라진다)."], "L374", "정확",
      r"C08\C08-T1\sensitivity.csv 12·13행, summary.txt 20행", "S2(한 출원인 ≤20% 상한) OLED +6.2→−0.4, 액정 −6.6 유지. 원안과 문자 동일. 단 '두 기업 70%'는 2017–19 출원 값(비교 창 2016–17과 다름, 서술상 허용)."),
 387: ([835], ["새 분쟁은 9/78건(과거 분쟁 기준선 4/78)이나 전향 채점에서는 4/65건(M0 7/65건)"], "L387", "정확",
      r"C07\C07-T3\verify_0\v02_out.json, subgroup_newdispute.csv", "전향 새 분쟁 65건 중 챔피언 4·M0 7(SB도 4). 원안과 문자 동일."),
 395: ([834], ["(포기·분쟁 위험 열은 공식 외 시험에서 단순 지속 지수보다 낫지 않았다)"], "L395", "정확",
      r"C07\C07-T3\ksic_rank_check.csv, C08\C08-T3\industry_rank_check.csv", "분쟁 ρ 0.513 대 지속 0.643, 포기 0.496 대 0.650(차이 CI 0 포함, FINDINGS 184·208). 원안과 문자 동일. 참고: 건수 기준(사전 등록 밖)에서는 분쟁 지수 0.718 > 지속 0.444."),
 534: ([850], ["| C13-SEED6 | 2.2, 2.3, 4.1, 4.4 | N4·N6·N7·N8·N9·T4 동결 스펙에서 학습 시드(N8·N9는 음성 표집 시드 포함)만 바꾼 8–10묶음"], "L534", "정확",
      r"C13\C13-T2\seed_bundles.csv, variance_decomp.csv", "묶음 10/10/10/9/9/8=56, CI 하한>0 56/56, 강화 분류 전 묶음 유지(T4 '조건부'), T4 강화 하한 −0.0158~−0.0025(→−0.016~−0.003), 시드 sd/부트스트랩 SE 0.055–0.192(→0.06–0.19). E-C13T2-01 대안 행과 문자 동일(출처 칸 포함)."),
}
out = []
for op, i, j in pairs:
    key = j if j is not None else i
    a = A.get(key)
    old = v14[i - 1] if i else ""; new = v15[j - 1] if j else ""
    present = [frag for frag in a[1] if frag in new] if a else []
    out.append(dict(v15_line=j, v14_line=i, op=op, errata_ids="; ".join(re.findall(r"(E-C1[23]T\d-\d+|E-C1[23]T\d-F\d)", " ".join(E(n)[:40] for n in a[0]))) if a else "",
                    errata_lines=",".join(map(str, a[0])) if a else "", 원안_핵심문구=" ‖ ".join(a[1]) if a else "", 원안문구_반영여부=f"{len(present)}/{len(a[1])}" if a else "",
                    반영문_v15=new[:600], 이전문_v14=old[:600], 판정=a[3] if a else "미분류", 재계산=a[2] if a else "", 근거파일=a[4] if a else "", 설명=a[5] if a else ""))
D = pd.DataFrame(out)
D.to_csv(W + r"\diff_audit.csv", index=False, encoding="utf-8-sig")
print("diff 출력 줄 수", n_diff_lines, "원고 줄 단위", len(D))
print(D[["v15_line", "v14_line", "op", "errata_ids", "원안문구_반영여부", "판정"]].to_string())

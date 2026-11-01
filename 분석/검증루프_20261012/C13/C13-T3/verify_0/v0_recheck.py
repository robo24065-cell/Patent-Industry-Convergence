# C13-T3 verify_0: 무작위 15개 주장(seed 20261017)을 FINDINGS가 아니라 원 결과 파일에서 직접 값을 꺼내 재판정한다.
import json, sys, csv, math
import pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
LP = B + r"\검증루프_20261012"; MC = B + r"\모델경쟁_20260927"
out = {}
# K036 (168) 재가중·S1
s8 = pd.read_csv(LP + r"\C08\C08-T1\sensitivity.csv")
r1 = pd.read_csv(LP + r"\C01\C01-T1\rta_sensitivity.csv")
p3 = r1[(r1.period == "P3") & (r1.countries == "5cc")]
out["K036"] = dict(reweight=p3[["method", "lead_KR", "kr_rta_gt1", "kr_leaders"]].to_dict("records"),
                   S1=s8[s8.item.str.contains("OLED|주방용|5개국 1위")][["item", "S1_main"]].to_dict("records"))
# K004 (13) R12 res_main + 시점
rm = pd.read_csv(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_main.csv")
st = pd.read_csv(LP + r"\C06\C06-T1\out\s4_time.csv")
out["K004"] = dict(main=rm[rm["기준"].isin(["NPS5", "DART5"])][["기준", "PROV-BEST", "PROV-BEST_lo", "PROV-BEST_hi", "PROV-BEST_w", "PROV-BEST_w_lo", "PROV-BEST_w_hi"]].to_dict("records"),
                   dart_2010=st[st["범위"] == "연도별"].iloc[[0, -1]][["문서구간", "PROV-BEST"]].to_dict("records"))
# K057 (253) T1 미래 물가 결측
fr = open(MC + r"\T1_산업대기실\R3\FINAL_REPORT_R3.md", encoding="utf-8").read()
out["K057"] = dict(price_missing_note=("c=2024에서는 물가 5피처가 결측" in fr))
# K024 (116) N5
n5 = json.load(open(MC + r"\N5_표준특허선언레이더\final_result_r4.json", encoding="utf-8"))
rb = open(MC + r"\N5_표준특허선언레이더\VERIFY_R5\robustness_table.md", encoding="utf-8").read()
out["K024"] = dict(skill=n5["skill_champ_vs_M0"], ci=n5["ci95_cluster_app"], strong_row=("**0.0002**" in rb))
# K085 (360) R8-P2 민감도
p2 = {k: json.load(open(LP + rf"\C07\C07-T1\cache\p2_{k}_results.json", encoding="utf-8"))["ship"]["compare"]["sig_PROV_KR-sig_TXT"] for k in ["B0", "V1", "V2a"]}
out["K085"] = p2
# K016 (30)
cr = json.load(open(MC + r"\_shared_report\audit_r7\concordance_resolution_check.json", encoding="utf-8"))
out["K016"] = dict(share_ipc_multi=cr["kr_2005_2022"]["share_ipc_multi_industry"], matched=cr["kr_2005_2022"]["matched_ipc"])
# K063 (269) N9·N6 비순환
nc = pd.read_csv(MC + r"\_shared_report\noncirc_n9n6_R9\results_table.csv")
out["K063"] = nc[(nc.comparison == "PROV-TXT")][["set", "mean_dAUC", "ci_lo", "ci_hi", "npos", "n_origins", "verdict"]].to_dict("records")
# K102 (408) 순환 판정
r10 = open(MC + r"\_shared_report\label_origin_R10\RESULT_R10.md", encoding="utf-8").read()
out["K102"] = dict(rule_unclear=("'불명'(경계)" in r10))
# K068 (287) N8 상위 1% 구성·보정
tl = json.load(open(LP + r"\C01\C01-T3\s04_toplist.json", encoding="utf-8"))
n8 = open(MC + r"\N8_특허분쟁표적레이더\FINAL_REPORT_R6.md", encoding="utf-8").read()
out["K068"] = dict(hits=tl["p_champ"]["hits"], prior=tl["p_champ"]["prior"], calib_204=("2.04배 과대" in n8))
# K067 (286) 무효 단독
rs = pd.read_csv(LP + r"\C01\C01-T3\rescoring_sensitivity.csv")
out["K067"] = rs[(rs.label.isin(["L_all", "L_inv"])) & (rs.metric.isin(["cap_champ", "cap_SB", "dcap_SB"]))][["label", "metric", "point", "ci_owner_lo", "ci_owner_hi"]].to_dict("records")
# K097 (395) 지수 대 지속
a = pd.read_csv(LP + r"\C07\C07-T3\ksic_rank_check.csv"); b = pd.read_csv(LP + r"\C08\C08-T3\industry_rank_check.csv")
out["K097"] = dict(N8_index=spearmanr(a.mean_p, a.realized_rate_permil).correlation, N8_persist=spearmanr(a.persist_rate_permil_2023_24, a.realized_rate_permil).correlation,
                   T4_index=spearmanr(b.index_pred_risk_3y, b.realized_rate_window).correlation, T4_persist=spearmanr(b.persist_rate_prev_window, b.realized_rate_window).correlation)
# K072 (300) T3 연계표 대체·과제 대체
es = pd.read_csv(MC + r"\T3_기술셀부상예측\ESSENTIAL_R7\essential_table_pub.csv")
out["K072"] = es[es.test.isin(["full", "conc_feat", "conc_task"])][["test", "mean_auc", "d_auc", "d_auc_ci95"]].to_dict("records")
# K071 (293)
rt = json.load(open(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_tests.json", encoding="utf-8"))
vd = json.load(open(LP + r"\C10\C10-T2\res\verdict.json", encoding="utf-8"))
out["K071"] = dict(A=rt["verdict"]["판정"], B_txtfree=vd["P50-FREE50"]["verdict"], B_two_way=vd["P50-FREE50"]["cluster_lo_gt0"])
# K043 (194) C2620
s7 = pd.read_csv(LP + r"\C07\C07-T1\sensitivity_313.csv", header=None)
out["K043"] = dict(S1=s8[s8.item.str.contains("C2620")][["item", "S0_reproduced", "S1_main", "S2_aux"]].to_dict("records"),
                   est=s7[s7[2].astype(str).str.contains("사업체수 연성장") & s7[2].astype(str).str.contains("2621")].iloc[:2, [2, 3, 4]].values.tolist())
# K076 (334)
bs = pd.read_csv(LP + r"\C10\C10-T2\res\boot_summary.csv")
r = bs[bs.pair == "P50-FREE50"].iloc[0]
out["K076"] = dict(mean=r.mean_d, firm=[r.firm_lo, r.firm_hi], firm_sc=[r.firm_sc_lo, r.firm_sc_hi], per_origin=r.per_origin)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=float)[:400])

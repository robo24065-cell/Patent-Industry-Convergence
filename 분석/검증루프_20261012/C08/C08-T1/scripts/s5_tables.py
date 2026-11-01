# -*- coding: utf-8 -*-
# 5단계: sensitivity.csv 조립 + 집중도 요약 (cache JSON·concentration_by_cell.csv만 읽음)
import sys, os, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import pandas as pd, numpy as np
r = json.load(open(os.path.join(CACHE, "s2_results_a2.json"), encoding="utf-8"))
r0 = json.load(open(os.path.join(CACHE, "s2_results_a0.json"), encoding="utf-8"))
w = json.load(open(os.path.join(CACHE, "s3_results.json"), encoding="utf-8"))
v = json.load(open(os.path.join(WD, r"verify_0\v_indep_results.json"), encoding="utf-8"))
rows = []
def add(line, item, orig, s0, s1, s2, verdict, note, s1_n0="", v0=""):
    rows.append(dict(manuscript_line=line, item=item, manuscript_value=orig, S0_reproduced=s0, S1_main=s1, S2_aux=s2, S1_key_N0=s1_n0, verify0_S1=v0, verdict_S1=verdict, note=note))
P = lambda k, v_: r[f"P3_{v_}"][k]
add(161, "KR RTA>1 산업 수(P3)", 25, P("kr_rta_gt1", "S0"), P("kr_rta_gt1", "S1"), P("kr_rta_gt1", "S2"), "영향 없음(Δ+1≤2)", "S1-KR 28, S2b 24", r0["P3_S1"]["kr_rta_gt1"], v["S1"]["kr_rta_gt1"])
add(161, "KR 5개국 1위 산업 수(P3)", 7, P("kr_leader", "S0"), P("kr_leader", "S1"), P("kr_leader", "S2"), "영향 없음(Δ+1≤1)", "원 7개 중 유지 5개(26212·28511 이탈, 15220·33301·33309 진입)", r0["P3_S1"]["kr_leader"], v["S1"]["kr_leader"])
add(161, "OLED 26212 KR RTA·1위", "1.68, KR 1위", f'{P("rta_26212","S0")} KR', f'{P("rta_26212","S1")} {P("lead_26212","S1")}(US 1.514)', f'{P("rta_26212","S2")} {P("lead_26212","S2")}',
    "수정 필요(원고가 이름 든 한국 1위 산업의 1위 판정 뒤집힘)", "KR 문서 1위 엘지디스플레이 35.4%, 2위 삼성디스플레이 34.3%; S1-KR 1.364 US; 상위 2개 제거 0.985(<1)",
    f'{r0["P3_S1"]["rta_26212"]} {r0["P3_S1"]["lead_26212"]}', f'{v["S1"]["oled_26212"]["KR"]} {v["S1"]["lead_26212"]}(US {v["S1"]["oled_26212"]["US"]})')
add(161, "무인 항공기 31312 KR RTA·1위", "1.61, KR 1위", f'{P("rta_31312","S0")} KR', f'{P("rta_31312","S1")} {P("lead_31312","S1")}', f'{P("rta_31312","S2")} {P("lead_31312","S2")}', "영향 없음", "KR 1위 출원인 몫 2.0%(유효 출원인 335)")
add(161, "화장품 20423 KR RTA·1위(원계열)", "1.72, KR 1위", f'{P("rta_20423","S0")} KR', f'{P("rta_20423","S1")} {P("lead_20423","S1")}', f'{P("rta_20423","S2")} {P("lead_20423","S2")}', "영향 없음", "KR 1위 출원인 몫 5.8%")
add(161, "주방용 전기기기 28511 KR 1위(원고 본문엔 이름 없음, 출처표 MAP-KR-TOP)", "1.30, KR 1위", "1.305 KR", "0.989 CN", "1.007 CN", "단서(본문 미명시)", "KR 문서 1위 엘지전자 47.6%")
p2 = lambda vv: r[f"P2_{vv}"]
add(172, "ρ 특허 RTA vs 수출 RCA(P2)", "0.356 [0.026, 0.625]", f'{p2("S0")["rho"]["rho"]} {p2("S0")["rho"]["ci"]}', f'{p2("S1")["rho"]["rho"]} {p2("S1")["rho"]["ci"]}', f'{p2("S2")["rho"]["rho"]} {p2("S2")["rho"]["ci"]}',
    "영향 없음(점추정이 원 CI 안)", "S1 CI 하한이 0 아래 — 원고가 이미 '정의에 민감(0.30–0.34, CI 하한 −0.04–0.01)'로 서술", f'{r0["P2_S1"]["rho"]["rho"]} {r0["P2_S1"]["rho"]["ci"]}', f'{v["S1"]["rho"]} {v["S1"]["rho_ci"]}')
add("176-181", "표 5 사분면 개수 SS/SW/WS/WW", "12/11/3/22", "12/11/3/22", "12/12/3/21", "12/13/3/20", "영향 없음", "SW 추가 25924; 원고 '정의에 따라 9–12개' 범위 안(S2 13은 밖)")
add("178-181", "표 5 산업 예 11개 사분면", "유지", "11/11", f'{11-len(p2("S1")["examples_flipped"])}/11', f'{11-len(p2("S2")["examples_flipped"])}/11', "영향 없음", "S2b(서로 다른 출원인 수)는 4개 뒤집힘(참고); N0 키 S2는 영상 기기 26519(경계) 1개")
s = lambda vv: r[f"S313_{vv}"]
add(191, "KR 출원 성장 코드 안 분산 비중", "56.1%", s("S0")["within_g_KR"], s("S1")["within_g_KR"], s("S2")["within_g_KR"], "영향 없음(|Δ|≤10%p)", "")
add(191, "한국 기원 점유 변화 코드 안 분산 비중", "72.8%", s("S0")["within_krorigin_share_change"], s("S1")["within_krorigin_share_change"], s("S2")["within_krorigin_share_change"], "영향 없음(|Δ|=5.2%p)", "S1-KR 58.1%(−14.7%p), S2b 42.6% — 보조 변형에서는 10%p 넘게 줄어듦")
add(191, "5개국 합계 성장 코드 안 분산 비중", "29.1%", s("S0")["within_g_5"], s("S1")["within_g_5"], s("S2")["within_g_5"], "영향 없음", "KR > 5개국 질적 주장 유지(모든 변형)")
add("194·9·374", "C2620 액정 26211 KR 출원 연평균 로그 성장", "−6.6%", s("S0")["gw_KR_C2620"]["26211"], s("S1")["gw_KR_C2620"]["26211"], s("S2")["gw_KR_C2620"]["26211"], "영향 없음(부호 유지)", "두 창 모두 1위 엘지디스플레이(39.8%→31.3%)", "", v["C2620"]["S1"]["26211"])
add("194·9·374", "C2620 OLED 26212 KR 출원 연평균 로그 성장", "+6.2%", s("S0")["gw_KR_C2620"]["26212"], s("S1")["gw_KR_C2620"]["26212"], s("S2")["gw_KR_C2620"]["26212"], "영향 없음(부호 유지, 크기 절반 이하)", "1위 엘지디스플레이(24.2%→42.2%); S2 −0.4·상위 2개 제거 −3.0으로 부호 뒤집힘(보조)", "", v["C2620"]["S1"]["26212"])
add(194, "메모리 26111 반례 KR 출원", "−14.8%", s("S0")["gw_KR_26111"], s("S1")["gw_KR_26111"], s("S2")["gw_KR_26111"], "영향 없음(부호 유지)", "", "", v["C2620"]["S1"]["26111"])
t3 = w["T3"]
add(223, "3.2.3 T3 예측 부상 비율 상위 10 KSIC 겹침", "10", "재현(최대차 1e-16)", t3["overlap_S1"], "", "영향 없음(≥7)", f'S1 제외 177셀(예측 9셀); 보조 S1b(어떤 출원인이든 셀 50% 이상) 겹침 {t3["overlap_S1b"]}')
add(223, "3.2.3 이름 든 4개(26112·28902·31312·27211) 비율", "21.9/19.6/19.5/18.8", "21.9/19.6/19.5/18.8", "/".join(str(t3["named_rates"][k]["S1"]) for k in ["26112", "28902", "31312", "27211"]), "", "영향 없음(상위 4 유지)", "S1b 20.8/19.4/19.0/15.7(27211 8위)")
t1 = w["T1"]
add(248, "3.3.3 T1 대기실 지수 상위 10 겹침·상위 5 비중", "8.6/6.9/6.2/5.1/3.5, 합 30.3%", "재현(지수 최대차 3e-5)", f'겹침 {t1["overlap_S1"]}, 합 {t1["top5_share_sum_S1"]}%', "", "영향 없음", f'KSIC 안 1위 법인 몫 중앙값 {t1["top1_firm_share_median"]}, 최대 {t1["top1_firm_share_max"]}')
for k in ["T4_R2", "T4_R3"]:
    add(395, f"4.3 포기 위험 산업 순위({k}, 원고에 산업 이름 없음)", "—", "재현(최대차 8e-17)", f'겹침 {w[k]["overlap_S1"]}, ρ {w[k]["rank_spearman_S0_S1"]}', "", "설계 단서 없음(≥7)", f'26211 1위→2위; 1위 권리자 몫 ≥20% KSIC 9개')
add(395, "4.3 분쟁 위험 산업 순위(N8 기대 분쟁 합, 원고에 산업 이름 없음)", "—", "재현(최대차 2e-7)", f'겹침 {w["N8"]["overlap_S1"]}', "", "설계 단서 없음", f'1위 권리자 기대분쟁 몫 중앙값 {w["N8"]["top1_holder_expdispute_share_median"]}, 최대 {w["N8"]["top1_holder_expdispute_share_max"]}(24191)')
S = pd.DataFrame(rows)
S.to_csv(os.path.join(WD, "sensitivity.csv"), index=False, encoding="utf-8-sig")
# 집중도 요약
C = pd.read_csv(os.path.join(WD, "concentration_by_cell.csv"), dtype={"ksic": str})
q = C[C.period.isin(["P2", "P3", "W0809", "W1617"])].groupby(["period", "cc"]).agg(top1_med=("top1_share", "median"), top1_max=("top1_share", "max"),
     top5_med=("top5_share", "median"), effn_med=("eff_n", "median"), n_top1_ge30=("top1_share", lambda x: int((x >= 0.3).sum())))
q.round(3).to_csv(os.path.join(CACHE, "s5_conc_summary.csv"), encoding="utf-8-sig")
print(S[["manuscript_line", "item", "S1_main", "verdict_S1"]].to_string())
print(q.round(3).to_string())
kr3 = C[(C.cc == "KR") & (C.period == "P3")].sort_values("top1_share", ascending=False).head(8)
print(kr3[["ksic", "top1_name", "top1_share", "top5_share", "eff_n"]].round(3).to_string())

# -*- coding: utf-8 -*-
"""사전 고정 명세(실행 전에 기록, 2026-10-04). 묶음 순서·하이퍼파라미터·원점·음성 표본비율은 결과를 보고 바꾸지 않는다.
채택: 평균 Δskill ≥ +0.003 AND 원점 60% 이상 우세(PROTOCOL_R2). SB = 강한 기준선(채택 판단 무관, 사후 보고용으로 사전 선언)."""

K1_M0 = ["ksic_i", "ksic2", "ipc_sec", "n_ipc", "nk", "util", "dom_prio", "n_app", "at_comp", "at_res", "at_univ",
         "wo_vis", "reg_by_t0", "reg_first", "lag_m", "vm"]
HIST = ["h_app_n", "h_app_rate", "h_app_rate_sm", "h_appks_n", "h_appks_rate_sm", "h_mg_n", "h_mg_rate_sm", "h_ks_rate_sm", "h_appwo_rate_sm"]
CTX = ["k_world_g", "k_kr_share", "k_kr_g", "k_cn_g", "k_cn_share", "k_us_share"]
TRADE = ["x_ex_l", "x_ex_g", "x_im_l", "x_im_g", "x_imcn_sh", "x_imcn_g", "x_net"]
SB_HIST = ["h_app_n", "h_app_rate_sm", "h_appwo_rate_sm"]

K2_M0 = ["cc_us", "cc_jp", "ksic_i", "ksic2", "ipc_sec", "n_ipc", "nk", "util", "dom_prio", "n_app", "at_comp", "at_res", "at_univ",
         "wo_vis", "n_oth_vis", "reg_by_t0", "reg_first", "lag_m", "vm"]

SPEC = {
    "K1": dict(panel="K1_panel.parquet", y="y", dev=list(range(2012, 2021)), min_train=2006, group="akey", neg_frac=0.3,
               base=K1_M0,
               steps=[("M1_출원인셀이력", HIST), ("M2_5개국KSIC맥락", CTX), ("M3a_무역KSIC", TRADE), ("M3b_NTIS기원", ["gov_rnd", "gov_firm", "gov_univ"])],
               sb={"SB_M0+출원인이력": K1_M0 + SB_HIST}),
    "K2": dict(panel="K2_panel.parquet", y="y", dev=list(range(2012, 2021)), min_train=2006, group="akey", neg_frac=0.06,
               base=K2_M0,
               steps=[("M1_출원인셀이력", HIST), ("M2_5개국KSIC맥락", CTX), ("M3a_무역KSIC", TRADE), ("M3b_ETSI선언", ["sep_decl"])],
               sb={"SB_M0+출원인이력": K2_M0 + SB_HIST}),
}

# ---- 2026-10-05 재시도에서 추가(실행 전 기록) ----
# 진단(d1_audit_diag.py → R6_diag_mechanical.csv): K1·K2는 't0까지 WO(PCT) 공개가 보이는가' 하나로 AUC 0.89/0.84,
# 여기에 공개 지연(lag_m)만 더하면 0.95/0.88이다. 양성의 83%/96%가 WO 공개 단위다. 결과가 이미 내린 PCT 출원 결정에서
# 기계적으로 따라 나오므로 '쉬운 과제'다. 그래서 진짜 의사결정인 'PCT 국내단계 진입'만 남긴 판(K1P·K2P)을 주 과제로 바꾼다.
#   K1P: K1 단위 중 wo_vis=1(t0까지 WO 공개, 해외 공개 없음) → t0+1..t0+2 US/JP/EP 공개(=국내단계 진입 공개).
#   K2P: K2 단위 중 wo_vis=1 → t0+1..t0+2 KR 공개(=한국 국내단계 진입).
# 묶음·하이퍼파라미터·원점은 K1·K2와 같다(비교 가능하게). M0에서 상수가 된 wo_vis만 뺀다. 음성 표본: K1P 없음, K2P 0.5.
K1P_M0 = [c for c in K1_M0 if c != "wo_vis"]
K2P_M0 = [c for c in K2_M0 if c != "wo_vis"]
SPEC["K1P"] = dict(panel="K1P_panel.parquet", y="y", dev=list(range(2012, 2021)), min_train=2006, group="akey", neg_frac=1.0,
                   base=K1P_M0,
                   steps=[("M1_출원인셀이력", HIST), ("M2_5개국KSIC맥락", CTX), ("M3a_무역KSIC", TRADE), ("M3b_NTIS기원", ["gov_rnd", "gov_firm", "gov_univ"])],
                   sb={"SB_M0+출원인이력": K1P_M0 + SB_HIST})
SPEC["K2P"] = dict(panel="K2P_panel.parquet", y="y", dev=list(range(2012, 2021)), min_train=2006, group="akey", neg_frac=0.5,
                   base=K2P_M0,
                   steps=[("M1_출원인셀이력", HIST), ("M2_5개국KSIC맥락", CTX), ("M3a_무역KSIC", TRADE), ("M3b_ETSI선언", ["sep_decl"])],
                   sb={"SB_M0+출원인이력": K2P_M0 + SB_HIST})

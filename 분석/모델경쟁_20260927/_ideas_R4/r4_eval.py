# -*- coding: utf-8 -*-
"""사전 등록 순차 실험 채점(M0 -> M1 -> M2 -> M3). 원점 v 평가, 학습 t0 <= v-2. 채택: 평균 Δskill >= +0.003 AND 원점 60% 이상 우세.
하이퍼파라미터 고정(r4_common.XGB_PARAMS), 튜닝 없음. 두 모형족(XGB 주, 로지스틱 보조) 병기."""
import sys, time, warnings
import pandas as pd, numpy as np
from r4_common import *
warnings.filterwarnings("ignore")

SETS = {
    "I6": dict(panel="I6_panel.parquet", y="y", dev=list(range(2014, 2021)), min_train=2011, group="okey", size="own_log", metric="strat",
               base=["age_rg", "age_ap", "rem_life", "claims", "n_inv", "n_ipc", "ipc_sec", "ksic2", "co_firm", "n_own", "is_univ",
                     "own_log", "own_past_n", "own_past_rate", "mg_past_rate", "mg_log", "paid_1y", "last_annl", "prior_comm"],
               steps=[("M1_NTIS_project", ["gov", "g_nproj", "g_univ", "g_gri", "g_firm", "g_dev", "g_basic", "g_funds", "g_sbiz", "g_endgap", "g_mind"]),
                      ("M2_cell_firm_demand", ["c_firm3", "c_firm_g", "c_firm_sh"]),
                      ("M3_KSIC_NTIS_funding", ["k_rnd_g", "k_rnd_sh"])]),
    "I6b": dict(panel="I6b_panel.parquet", y="y", dev=list(range(2014, 2021)), min_train=2011, group="okey", size="own_log", metric="strat",
                base=["age_rg", "age_ap", "rem_life", "claims", "n_inv", "n_ipc", "ipc_sec", "ksic2", "co_firm", "n_own", "is_univ",
                      "own_log", "own_past_n", "own_past_rate", "mg_past_rate", "mg_log", "paid_1y", "last_annl", "prior_comm"],
                steps=[("M1_inventor_history", ["inv_np_max", "inv_cm_max", "inv_cm3_max", "inv_cm_rate", "inv_any_cm"]),
                       ("M2_NTIS_project", ["gov", "g_nproj", "g_univ", "g_gri", "g_firm", "g_dev", "g_basic", "g_funds", "g_sbiz", "g_endgap", "g_mind"]),
                       ("M3_cell_firm_demand", ["c_firm3", "c_firm_g", "c_firm_sh"])]),
    "I7": dict(panel="I7_panel.parquet", y="y", dev=[2017, 2018, 2019, 2020], min_train=2015, size="log_emp", n_boot=300,
               base=["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type"],
               steps=[("M1_patent_behavior", ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "lapse_1y", "lapse_2y",
                                              "pay_1y", "pay_trend", "trf_2y", "live_ratio"]),
                      ("M2_NTIS_participation", ["g_part_any", "g_part_n", "g_part_proj", "g_part_univ", "g_part_gri", "g_part_firm", "g_part_rec", "g_part_share"]),
                      ("M3_KSIC_NTIS_funding", ["k_rnd_g", "k_rnd_sh"])]),
    "I8": dict(panel="I8_panel.parquet", y="y", dev=list(range(2014, 2020)), min_train=2012, size="n_proj", n_boot=300,
               base=["n_proj", "funds", "n_yrs", "rec", "sbiz_sh", "cont_sh", "sme", "career", "n_proj_all", "corp_type"],
               steps=[("M1_project_content_KSIC50", ["ph_dev", "ph_app"] + ["t6_" + x for x in ["IT", "BT", "NT", "ET", "ST", "CT"]]
                       + ["cl_" + x for x in ["기계", "전기전자", "정보통신", "화공", "재료", "보건의료", "농림수산", "생명과학"]] + ["k50_assigned", "k50_score", "b1_score"]),
                      ("M2_NPS", ["nps_present", "log_emp", "g_emp1", "nps_age"]),
                      ("M3_KSIC_NTIS_funding", ["k_rnd_g"])]),
}


def main(idea, extra_tag=""):
    """I6 주 지표 = 기관 내 AUC(strat). 1차 실행(전체 풀 AUC)에서 기관 간 사업화율 변동이 원점마다 크게 흔들려(M0 AUC 0.49~0.69)
    TLO 실사용(자기 포트폴리오 안에서 순위)에 맞춰 바꿨다 — 설계 변경은 IDEAS_R4에 공개. 풀 AUC도 함께 기록."""
    S = SETS[idea]
    D = pd.read_parquet(f"{CACHE}/{S['panel']}")
    D = D[D[S["y"]] >= 0].copy()
    assert D.t0.max() + 2 <= 2022, "결과창이 2022-12-31 뒤로 가는 원점 금지"
    t = time.time()
    R, O, preds = run_steps(D, S["y"], S["dev"], S["base"], S["steps"], group_col=S.get("group"), min_train=S.get("min_train"), size=S.get("size"),
                            metric=S.get("metric", "pooled"), n_boot=S.get("n_boot", 200))
    R.insert(0, "idea", idea); O.insert(0, "idea", idea)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False)); print(O.round(4).to_string(index=False)); print("sec", round(time.time() - t))
    R.to_csv(f"{OUT}/experiments_{idea}{extra_tag}.csv", index=False, encoding="utf-8-sig")
    O.to_csv(f"{OUT}/pilot_by_origin_{idea}{extra_tag}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(sys.argv[1])

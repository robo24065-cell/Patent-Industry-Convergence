# -*- coding: utf-8 -*-
"""사전 고정 실험 명세(묶음·순서·원점). r5_eval.py와 r5_robust.py가 같이 읽는다."""
SPEC = {
    "J3": dict(panel="J3_panel", dev=list(range(2017, 2021)), gap=2, n_boot=300,
               base=["nps_present", "log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "corp_type"],
               steps=[("M1_특허행태", ["log_app", "n_ksic", "yrs_since_last", "app_trend", "known_1y", "log_live", "live_ratio", "lapse_1y", "lapse_2y", "pay_1y", "pay_trend", "trf_2y"]),
                      ("M2_공동권리관계", ["co_corp_n", "co_corp_share", "co_top_port", "co_top_ratio", "past_acquirer"]),
                      ("M3_NTIS", ["ntis_n", "ntis_funds", "ntis_any"])]),
    "J1": dict(panel="J1_panel", dev=list(range(2014, 2021)), gap=2, neg_frac=0.3, group="org",
               base=["log_gov", "log_tot", "nongov_sh", "sbiz_sh", "perform_grp_c", "dev_phase_c", "cont", "series_yr", "total_len", "period_m", "end_in_year", "ministry_c", "has_pre",
                     "org_nproj", "org_kr_pat", "org_kr_rate", "pre_kr_pat"],
               steps=[("M1_과제내용", ["b1_tfidf_score", "b1_tfidf_assigned", "b2_dict_score", "b2_dict_assigned", "sixt_cd_c", "cls_L_canon_c", "econ_goal_cd_c", "tech_lifecycle_c", "apply1_cd_c"]),
                      ("M2_제공50산업이력", ["org_p50", "org_p50_rate", "org_p50_share", "pre_p50"])]),
    "J2": dict(panel="J2_panel", dev=list(range(2012, 2021)), gap=2, neg_frac=0.05, group="owner",
               base=["age_rg", "age_ap", "rem_life", "claims", "n_ipc", "n_inv", "utility", "atype", "foreign", "n_appl", "ksic_i", "ipc_sec", "last_annl", "paid_ahead"],
               steps=[("M1_이력", ["own_disp", "own_trial", "tech_eval", "n_trf", "trf_2y", "merger_trf", "lic", "owner_live", "owner_disp5", "mg_disp5", "ksic_disp5", "fam_offices", "fam_us"]),
                      ("M2_외부단위", ["us_granted", "us_sci", "us_papercit", "kpss_xi", "etsi", "gov", "gov_firm", "gov_univ", "gov_gri"])]),
}


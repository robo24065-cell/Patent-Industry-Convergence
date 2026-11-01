import sys, json; sys.stdout.reconfigure(encoding="utf-8")
W = sys.argv[1]
s = json.load(open(W + "/_s4_raw.json", encoding="utf8"))
ci = s["N7_official_ci"]; r = s["N7_rescore"]
wa = ci["auc"][1] - ci["auc"][0]; ws = ci["skill_M0"][1] - ci["skill_M0"][0]; wsb = ci["skill_SB"][1] - ci["skill_SB"][0]
off = r["official_frozen"]; cor = r["alt_firm_corr"]; wr = r["alt_firm_wrong"]
n7 = {
 "task": "C03-T3 (C) N7 perform_grp 매핑 오류 민감도 — 재학습 없음, 동결 예측 final_test_pred_r5.parquet 재채점",
 "mapping_check": {"source": "ntis_project_raw perform_agent_cd × perform_agent(텍스트) 교차표",
   "rows": s["code_text_grp"], "conclusion": "코드 06 텍스트='정부부처'(1,973행)인데 perform_grp='중견기업', 코드 08 텍스트='중견기업'(10,456행)인데 '기타' → ntis_B_build_raw.py GRP 매핑 오류 확인. 교정: 06→정부부처, 08→중견기업"},
 "code_path": {"population_and_label": "n7lib.build_panel: first_ntis = ntis_first_year(NT) = 과제연도 2010~의 corp_rgst_no(사업자번호) 첫 연도. perform_grp는 load_ntis에서 읽지만 사용하지 않음",
   "features": "M2 g_part_firm은 ntis_patent_link.perform_grp(ntis_rpatent_build.agent_grp: 04·05·08→기업, 06→정부부처; 올바른 매핑) 사용; M3 k_rnd_g·k_rnd_sh는 ntis_project_ksic_year의 gov_funds 합계(pg3 열 미사용). ky_xw.py의 FIRM/pg3는 ESSENTIAL_R7 대체 시험용 gov_funds_firm 열에만 쓰이며 공식 모형 피처가 아님",
   "conclusion": "공식 N7 모집단·레이블·피처는 raw perform_grp 매핑과 무관"},
 "final_origin_2022": {"n": s["N7_final"]["n"], "pos_frozen": s["N7_final"]["pos_frozen"], "pos_reconstructed": s["N7_final"]["pos_recon"],
   "label_match_rate": s["N7_final"]["label_match"], "population_violation": s["N7_final"]["pop_violation_first_le_2022"],
   "first_year_identical_after_correction": s["N7_final"]["official_first_year_identical_after_correction"],
   "positives_change_official_definition": 0, "positives_change_share": 0.0,
   "population_change": 0,
   "positives_entry_code_mix": s["N7_final"]["positives_entry_code_mix"],
   "note": "양성 320곳 중 16곳(5.0%)은 첫 진입 과제가 코드 08(실제 중견기업)뿐 — 공식 정의는 이들을 정상 포함"},
 "rescore_official_corrected": {"auc_champ": off["auc_champ"], "auc_M0": off["auc_M0"], "auc_SB": off["auc_SB"], "skill_vs_M0": off["skill_vs_M0"], "skill_vs_SB": off["skill_vs_SB"],
   "top1pct_hits_champ_M0": [off["champ_top123_hits"], off["M0_top123_hits"]], "top5pct": [off["champ_top614_hits"], off["M0_top614_hits"]], "top10pct": [off["champ_top1229_hits"], off["M0_top1229_hits"]],
   "delta_auc": 0.0, "delta_skill_M0": 0.0, "delta_skill_SB": 0.0},
 "official_ci": ci, "ci_width": {"auc": wa, "skill_M0": ws, "skill_SB": wsb, "quarter_auc": wa / 4, "quarter_skill_M0": ws / 4},
 "reference_alt_definition_firm_type_first_entry": {
   "desc": "비공식 참고: '기업 유형(대·중견·중소) 과제로 처음 진입'으로 정의했다면. 틀린 매핑에서는 코드 08 진입 16곳이 양성에서 빠짐",
   "wrong_mapping": wr, "corrected_mapping": cor,
   "wrong_vs_corrected_positive_sym_diff": r["alt_wrong_vs_corr_sym_diff"],
   "wrong_mapping_positive_change_share": 16 / 320,
   "wrong_mapping_delta_vs_official": {"auc": wr["auc_champ"] - off["auc_champ"], "skill_M0": wr["skill_vs_M0"] - off["skill_vs_M0"], "skill_SB": wr["skill_vs_SB"] - off["skill_vs_SB"]}},
 "dev_origins_positive_rate": s["N7_dev_origins"],
 "dev_summary": "공식 정의 재구성은 개발 원점 6개(2015–2020) 모두 동결 패널 y와 100% 일치(교정 전후 동일). 대안 정의에서는 틀린 매핑이 교정 매핑보다 양성률을 0.20–0.43%p 낮춤(교정 매핑 대안 정의는 공식과 0.01–0.05%p 차)",
 "prereg_verdict": "영향 없음 (양성 변동 0% < 2%, AUC·skill 변화 0 < CI 폭 1/4(AUC 0.0137, skill 0.0383), 기준선 우위 판정 불변)",
 "implication": "E-C02-D3(ntis_B_build_raw.py 매핑 교정)은 N7 공식 성적에 영향 없이 반영 가능. 다만 연례 재학습 때 N7 정의를 '기업 유형 과제'로 바꾸면 매핑 교정이 필수(틀린 매핑이면 양성 5.0% 누락)"
}
json.dump(n7, open(W + "/n7_grp_impact.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
n9 = s["N9"]
fw, fc = n9["firm_wrong"], n9["firm_corr"]
out9 = {"task": "C03-T3 (C) N9 '기업' 하위집단 교정 재채점 — final_preds_r6.parquet(동결) × ntis_project_raw(year=2022) perform_agent_cd",
 "test_n": n9["n"], "unmatched": n9["unmatched"], "secret_projects_in_test": n9["secret_in_test"],
 "official_reproduced": fw, "official_reported": n9["official_subgroup_firm"],
 "corrected_firm_codes_04_05_08": fc, "moved_in_code08": n9["moved_in_code08"], "moved_in_pos": n9["moved_in_pos"],
 "moved_out_code06": n9["moved_out_code06"], "moved_out_pos": n9["moved_out_pos"], "code08_only_subgroup": n9.get("code08_only"),
 "delta": {"auc": fc["auc"] - fw["auc"], "skill": fc["skill"] - fw["skill"], "top1": fc["top1"] - fw["top1"]},
 "note_metric": "원고 '기업 0.257'은 하위집단 skill(vs M0_L2)이다. 하위집단 AUC는 0.833(교정 0.838)",
 "prereg_verdict": "영향 없음 (하위집단 AUC 변화 +0.005 < 0.05, 0.5 기준 뒤집힘 없음; skill 0.257→0.259)",
 "caveat": "N9 피처 perform_grp_c·minper_c·firm도 같은 틀린 매핑으로 학습·예측됨(코드 08이 99와 함께 '기타'로 합쳐짐). 학습·시험이 같은 매핑이라 누수는 없으나, 교정 매핑 효과는 재학습 없이 측정 불가(금지) — 연례 재학습 시 반영"}
json.dump(out9, open(W + "/n9_sub_recheck.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(n7["prereg_verdict"]); print(out9["prereg_verdict"], out9["delta"])

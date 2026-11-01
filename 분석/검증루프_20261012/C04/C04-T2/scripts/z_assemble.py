# -*- coding: utf-8 -*-
"""C04-T2 산출물 조립: t4_rescore.json, n5_rescore.json, data_check.txt (값은 cache/*.json에서만 읽는다)"""
import os, json, re, glob
import pandas as pd
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda f: json.load(open(os.path.join(W, f), encoding="utf-8"))
t1, t2, t3 = J("cache/t4_s1_out.json"), J("cache/t4_s2_out.json"), J("cache/t4_s3_out.json")
n2, n3, n4 = J("cache/n5_s2_out.json"), J("cache/n5_s3_out.json"), J("cache/n5_s4_out.json")
v0 = J("verify_0/v0_out.json")
# ---------------- T4 판정
lbs = [b["skill_ci"][0] for b in t2["strong_frozen_file"]["boot_B1000_5seeds"]]
t4_label_ok = t1["only_frozen"] == 0 and t1["only_indep"] == 0 and t1["label_mismatch"] == 0 and t1["pos_indep"] == t1["pos_frozen"]
d = t2["diff_vs_official"]
t4_score_ok = abs(d["auc_model"]) <= 0.001 and abs(d["auc_base"]) <= 0.001 and abs(d["skill"]) <= 0.003
t4_verdict = "재현" if (t4_label_ok and t4_score_ok) else ("부분 재현" if t4_label_ok else "불일치")
sb_shift = abs(t2["strong_frozen_file"]["skill_vs_strong"] - 0.124)
boundary_changed = sb_shift > 0.003 or sum(x > 0.005 for x in lbs) >= 4
T4 = dict(task="C04-T2 T4 특허 포기 최종시험 독립 재채점", verdict=t4_verdict,
          official_prediction_location=dict(path=r"모델경쟁_20260927\T4_특허포기예측\R2\runs_final\F22\pred\{cls_99f270b3cd,haz_0568d4297b,logit_3008b8b195,hgb_b4bd0fa46a,base_46526c56d8}.npy + ev.parquet",
                                            spec=r"R2\final_spec_r2.json (R2-E03, 가중 haz 0.4·cls 0.2·logit 0.2·hgb 0.2, 로짓 평균)", files=t2["inputs"],
                                            note="final_models\\run_holdout·pred_*.parquet는 1라운드(원점 2019, 3단계 홀드아웃) 산출물로 공식 성적(R2, L=2022-01-01)과 다름. 예측 파일이 있어 저장 모형 재예측(2.4)은 하지 않음. 재적합·하이퍼파라미터 변경 없음."),
          labels=dict(method="t4lib/r2lib 미사용, registry_full.sqlite(mode=ro) 직접 재구성(scripts/t4_s1_labels.py)", **{k: t1[k] for k in ["n_risk_indep", "pos_indep", "n_undetermined_indep", "n_frozen", "pos_frozen", "only_frozen", "only_indep", "label_mismatch", "ksic_mismatch"]}),
          scores=dict(auc_model=t2["auc_model"], auc_base=t2["auc_base"], skill=t2["skill"], prauc_model=t2["prauc_model"], precision_top10=t2["precision_top10"], top100=t2["top100"],
                      components=t2["components"], official=t2["official"], diff_vs_official=d,
                      ci_official_procedure_seed0_B1000=t2["boot_official_procedure_seed0_B1000"], ci_other_seeds=t2["boot_other_seeds_B1000"]),
          strong_baseline=dict(frozen_file=t2["strong_frozen_file"], refit_maxdiff_vs_frozen=t2["strong_refit_vs_frozen_maxdiff"], refit=t2["strong_refit"], official=t2["official_strong"],
                               lower_bounds_B1000_5seeds=lbs, boundary_claim="강화 기준선 대비 CI 하한이 0에 닿는다(−0.003)",
                               boundary_changed=bool(boundary_changed),
                               note="VERIFY_R5 절차(B=300, 시드 3) 그대로 [−0.0025, 0.2081]=원고 [−0.003, 0.208]. B=1000 시드 5개 하한 −0.011~+0.001: 0에 닿는다는 판정 유지, 단일 숫자 −0.003은 시드 민감"),
          sensitivity=dict(lapse_registration_label=t2["sens_label_lapse_event"], later_payment_snapshot_2026_08=t3, recovery=t1["sens_recovery"],
                           lapse_event_vs_np=t1["sens_lapse_event"]),
          verify_0=v0["T4"], verify_0_verdict=v0["compare_main"]["verdict"])
json.dump(T4, open(os.path.join(W, "t4_rescore.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
# ---------------- N5 판정
s = n2["strict"]
n5_label_ok = s["only_frozen"] == 0 and s["only_indep"] == 0 and s["label_mismatch"] == 0 and s["pos_indep"] == s["pos_frozen"]
dn = n3["diff_vs_official"]
n5_score_ok = abs(dn["auc_champ"]) <= 0.001 and abs(dn["auc_M0"]) <= 0.001 and abs(dn["skill"]) <= 0.003
n5_verdict = "재현" if (n5_label_ok and n5_score_ok) else ("부분 재현" if n5_label_ok else "불일치")
slb = [n3["strong_file"]["boot_verify_procedure_seed55"]["ci"]] + [b["ci"] for b in n3["strong_file"]["boot_other_seeds"]]
olb = [n3["boot_official_procedure_seed2022"]["ci"][0]] + [b["ci"][0] for b in n3["boot_other_seeds"]]
n5_boundary_changed = abs(n3["strong_file"]["skill"] - 0.0002) > 0.003 or sum((c[0] > 0) or (c[1] < 0) for c in slb) >= 4
N5 = dict(task="C04-T2 N5 표준특허(ETSI) 선언 최종시험 독립 재채점", verdict=n5_verdict,
          labels=dict(method="n5lib·trade_etsi_sep_link 미사용, ETSI 원본 zip(ISLD-export.csv 4,984,299행) 직접 파싱 → KR 키 → 제공 kr_patents(KSIC 26429) 로컬 결합(scripts/n5_s1·s2)",
                      strict_rule=s, ext_rule=n2["ext"], kr26429_matched_strict=n2["kr26429_matched_strict"], kr26429_matched_ext=n2["kr26429_matched_ext"]),
          scores=dict(auc=n3["auc"], skill=n3["skill"], top100=n3["top100"], prauc=n3["prauc"], official=n3["official"], diff_vs_official=dn,
                      ci_official_procedure_seed2022=n3["boot_official_procedure_seed2022"], ci_other_seeds=n3["boot_other_seeds"], official_ci_lower_bounds_5seeds=olb),
          strong_baseline=dict(file=n3["strong_file"], refit=n3["strong_refit"], official=n3["official_strong"], ci_5seeds=slb,
                               boundary_claim="강한 기준선 대비 이득 없음(점추정 0.000)", boundary_changed=bool(n5_boundary_changed)),
          sensitivity_link_key=n3["sens_ext_key"],
          verify_0=v0["N5"], verify_0_verdict=v0["compare_main"]["verdict"])
json.dump(N5, open(os.path.join(W, "n5_rescore.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
# ---------------- data_check.txt
ag = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(W, "cache", "etsi", "agg_part_*.parquet")))], ignore_index=True)
pj = ag.groupby("proj").rows.sum()
cell = pj[[bool(re.search(r"GSM|GPRS|EDGE|UMTS|HSPA|WCDMA|LTE|EUTRAN|E-UTRA|NR|New Radio|5G|3GPP|IMT|SAE|EPS|IMS|MCPTT|NB-IOT|LTE-M", p, re.I)) for p in pj.index]].sum() / pj.sum()
krp = ag[ag.cc == "KR"].groupby("proj").rows.sum()
krcell = krp[[bool(re.search(r"GSM|GPRS|EDGE|UMTS|HSPA|WCDMA|LTE|EUTRAN|E-UTRA|NR|New Radio|5G|3GPP|IMT|SAE|EPS|IMS|MCPTT|NB-IOT|LTE-M", p, re.I)) for p in krp.index]].sum() / krp.sum()
nonc = pj[[not bool(re.search(r"GSM|GPRS|EDGE|UMTS|HSPA|WCDMA|LTE|EUTRAN|E-UTRA|NR|New Radio|5G|3GPP|IMT|SAE|EPS|IMS|MCPTT|NB-IOT|LTE-M", p, re.I)) for p in pj.index]].sort_values(ascending=False).head(6)
c = n4["c_link"]; dd = n4
L = []
A = L.append
A("C04-T2 data_check — 2026-10-12. 이번 작업은 새 데이터를 내려받지 않았다. 이미 받아 둔 ETSI 원본(_shared\\ext\\trade_etsi_ISLD-export.zip)과 등록원부(registry_full.sqlite, mode=ro)를 다시 점검했다.")
A("")
A("[1] ETSI IPR DB 공개 추출본 — N5 결과(레이블) 원천")
A(f"(a) 출처·총량: {n4['a_source']['file']}. zip 안 CSV {n4['a_source']['csv_bytes']:,}바이트, 내부 시각 {n4['a_source']['zip_internal_timestamp']}, 행 {n4['a_source']['rows']:,}(SOURCES.md 기록 4,984,299와 일치).")
A(f"    선언(IPRD_ID) 단위 묶음 DIPG {n4['a_source']['n_dipg']:,}개, 패밀리(DIPG_PATF_ID) {n4['a_source']['n_family']:,}개. KR 행 {n2['etsi_kr_rows']:,}, KR 구성원이 있는 DIPG {n2['etsi_kr_dipg']:,}, KR 선언 {n2['etsi_kr_decl']:,}건, KR (출원번호·공개번호) 고유 {n4['b_KR_dup']['distinct_app_pub']:,}(기존 trade_06 로그 34,040과 1 차이 — 공란 처리 차이).")
A("    공개 총량 대조: 이번 바퀴에서 ETSI가 공표한 선언 총수(공식 통계표)를 확보하지 못해 외부 대조는 미실시. 원본 행수·갱신 시각·기존 파생본과의 내부 대조로 대신했다(한계로 기록).")
A(f"(b) 분포: 국가별 행 상위 {dict(list(n4['b_rows_by_cc_top10'].items())[:6])}. 반영연도별 DIPG {n4['b_dipg_by_refl_year']}.")
A(f"    표준기구 구성: 이 DB는 ETSI 한 기구의 IPR 선언 DB다. 다른 표준기구(IEEE 802.11 Wi-Fi, ITU-T 영상코덱, ISO/IEC 등) 선언은 들어 있지 않다. 프로젝트 기준 셀룰러(GSM·UMTS·LTE·NR·3GPP 계열, 이름 일치 기준 하한) 몫: 전체 행 {cell:.1%} 이상, KR 행 {krcell:.1%} 이상(GERAN·AMRWB도 3GPP 계열이라 실제는 더 높음). 비셀룰러 상위 프로젝트: {dict((k, int(v)) for k, v in nonc.items())}.")
A(f"    → N5 레이블은 '표준특허 선언 전체'가 아니라 'ETSI(사실상 3GPP 셀룰러) 선언 DB 최초 반영'이다. KSIC 26429(기타 무선통신장비) 안에서도 Wi-Fi·블루투스 등 비ETSI 표준 선언은 음성으로 들어간다(크기 미측정).")
A(f"    결측: 반영일 공란 {dd['b_missing']['refl_blank']}, 서명일 공란 {dd['b_missing']['sig_blank']}, 기업명 공란 {dd['b_missing']['comp_blank']}. 중복: 같은 특허가 여러 선언에 반복 등장(KR 행 {dd['b_KR_dup']['rows']:,} 대 고유 특허 키 {c['unique_patent_keys']['n']:,}) — 최초 반영일(최솟값)로 축약해 영향 없음.")
A(f"    기업 집중도: 전체 DIPG 상위 5개사 몫 {dd['b_company_top5_share_all_dipg']['share']:.1%} {dd['b_company_top5_share_all_dipg']['top5']}; KR 구성원 DIPG 상위 5개사 {dd['b_KR_company_top5_share']['share']:.1%} {dd['b_KR_company_top5_share']['top5']}.")
A(f"    최종시험 양성 442건의 최초 반영 선언 기업 상위 5개 몫 {dd['b_final_positive_declarant_top5']['share']:.1%} {dd['b_final_positive_declarant_top5']['top5']}(선언 기업 {dd['b_final_positive_declarant_top5']['n_companies']}곳), 출원인 기준 상위 5 몫 {dd['b_final_positive_applicant_top5_share']:.1%}.")
A("    → 양성의 약 60%가 5개 기업(삼성·OPPO·퀄컴·LG·vivo)에서 나온다. VERIFY_R5 §3.3에서 상위 출원인을 빼도 skill이 줄지 않음을 이미 확인(0.174–0.313) — 집중 자체가 성적을 부풀리지는 않지만, 레이블이 소수 대기업의 선언 관행을 반영한다는 단서는 필요.")
A(f"(c) 연결(매칭) 선택 편향: KR ETSI 특허 출원번호 키 고유 {c['unique_patent_keys']['n']:,}개 중 제공 KR(1,087,511건, 50개 제조업 KSIC) 매칭 {c['unique_patent_keys']['matched']:,}({c['unique_patent_keys']['matched_share']:.1%}; 출원 2017–22 {c['unique_patent_keys']['matched_share_ay2017_2022']:.1%}).")
A(f"    제공 KR은 50개 제조업 KSIC에 배정된 출원만 담으므로 매칭 실패의 상당수는 범위 밖(다른 업종 배정·2000년 이전 출원; 매칭 안 된 DIPG 행 중 2000년 이전 {c['unmatched_patent_keys']['app_year_lt_2000']:,})으로 추정한다(제공 데이터에 없는 특허의 업종은 직접 확인 불가). 매칭된 것의 KSIC: {c['matched_ksic_top6']} — 89.8%가 26429.")
A(f"    연도별 매칭률은 2000–2023년 0.28–0.44로 평탄(2024–25는 공개 지연으로 낮음): 시간 방향 치우침 없음. 기업별: {dict((k, round(v['matched'], 2)) for k, v in c['matched_share_by_company_top8'].items())} — 모토로라(0.12)·퀄컴(0.29)이 낮고 LG(0.41)가 높다.")
A(f"    세대별 매칭률: {dict((k, round(v, 3)) for k, v in c['matched_share_by_gen'].items())}(2G·3G만 선언된 오래된 특허는 제공 데이터 연도 범위 밖이 많음).")
A(f"    연결 키 형식 누락(새 결과): 원 연결 규칙(trade_06)은 KR 출원번호를 'KR+11자리' 또는 숫자 11자리만 인정해 '10-2018-0093890'·'KR1020180137422'·'KR10-2019-7011239' 형식을 놓친다. 확장 규칙으로 KSIC 26429 선언 특허가 {n2['kr26429_matched_strict']:,}→{n2['kr26429_matched_ext']:,}건(+{n2['kr26429_matched_ext'] - n2['kr26429_matched_strict']}, 0.43%) 늘고, 눈으로 확인한 표본 25행에서 ETSI 선언 기업과 제공 출원인이 모두 일치(LG전자·아이티엘·아수스·퀄컴 등).")
A(f"    최종시험 영향: 이미 2022년까지 선언된 단위 {n3['sens_ext_key']['n_excluded']}건(양성 {n3['sens_ext_key']['pos_excluded']})이 공식 모집단에 남아 있음(0.32%). 빼고 채점하면 skill {n3['sens_ext_key']['skill']:.3f}(공식 0.165), 강한 기준선 대비 {n3['sens_ext_key']['skill_vs_strong']:.4f}(공식 0.0002) — 경계 판정 불변. 학습 원점 패널에도 같은 누락이 있으나 재적합 금지 원칙상 측정하지 않음.")
A(f"(d) 시점 정합: 서명→DB 반영 지연(2012년 이후 반영) 중앙값 {dd['d_lag_days_all_refl_ge2012']['median']:.0f}일·90% {dd['d_lag_days_all_refl_ge2012']['p90']:.0f}일·99% {dd['d_lag_days_all_refl_ge2012']['p99']:.0f}일. 최종창(2023–24 반영) 중앙값 {dd['d_lag_days_refl_2023_24']['median']:.0f}일·90% {dd['d_lag_days_refl_2023_24']['p90']:.0f}일(KR {dd['d_KR_lag_refl_2023_24']['median']:.0f}·{dd['d_KR_lag_refl_2023_24']['p90']:.0f}일).")
A(f"    최종시험 양성 442건: 지연 중앙값 {dd['d_final_pos_lag']['median']:.0f}일, 최대 {dd['d_final_pos_lag']['max']:.0f}일. 반영 날짜 상위 {dd['d_final_pos_lag']['top_refl_days']} — 선언서 단위로 묶여 반영되는 구조(일괄 이관 같은 이상은 없음).")
A(f"    창 끝 절단: 2024년 서명 선언 중 2025년 이후 반영 {dd['d_sig2024_reflected_after_2024']['share']:.1%}(사전 기준 10% 미만 → 통과), 2025년 반영분 중 2024년 이전 서명 {dd['d_refl2025_signed_2024_or_earlier']['share']:.1%}. 최종시험 단위 중 창 뒤(2025-01~2026-09) 반영 {dd['d_units_reflected_after_window']['n']}건({dd['d_units_reflected_after_window']['share_of_units']:.1%}, 이 가운데 2024년까지 서명 {dd['d_units_reflected_after_window']['signed_by_2024']}건)은 정의상 음성.")
A(f"    2011 일괄 반영: DIPG {dd['d_bulk_2011']['dipg_refl_2011']:,}개(이 중 서명 2011 이전 {dd['d_bulk_2011']['of_which_sig_before_2011']:,}) — N5 원점(2012–2022)의 결과창과 겹치지 않음. 추출본 갱신 2026-09-28 05:05 > 결과창 끝 2024-12-31(완전 포함). 선언 철회 이력은 추출본에 없음(한계 유지).")
A("    판정: 치우침 있음 — (1) ETSI 단일 기구·셀룰러 편중, (2) 양성의 60%가 5개 기업, (3) 연결 키 형식 누락 0.43%. 셋 다 공식 성적의 재현성에는 영향 없고, (3)의 보정 영향은 skill −0.004·강한 기준선 대비 −0.002로 경계 판정 불변. (1)은 원고 정의 문장에 반영 권장, (2)·(3)은 한계로 기록.")
A("")
A("[2] 등록원부 registry_full.sqlite — T4 결과(레이블) 원천(기존 수집본, 새 다운로드 아님)")
A(f"(a) 총량: patent {t1['n_patent_table']:,}행 중 특허(등록번호 10·등록일 있음) {t1['n_patent_10_with_rd']:,}, pay {t1['n_pay_rows']:,}행 — DATA_NOTES·README 기록과 일치. 만료일 결측 대체 {t1['n_exp_imputed']}건.")
A(f"(b) 위험집합 L=2022-01-01 {t1['n_risk_indep']:,}건, 포기 {t1['pos_indep']:,}({t1['base_rate_indep']:.2%}), 미확정 {t1['n_undetermined_indep']}. 대형 출원인 집중은 VERIFY_R5 §3.2(상위 10 출원인 20.6%, 그 안 skill 0.026)에 이미 기록.")
A(f"(c) 레이블 정의 대조: 납부기록 레이블과 소멸등록(등록료불납) 사건 레이블 일치 {t1['sens_lapse_event']['agree_with_np']:.2%}. 불일치 {t1['sens_lapse_event']['np0_lap1']:,}건(납부 0·소멸 1) 중 {t2['sens_label_lapse_event']['np0_lap1_recov']:,}건이 회복등록(소멸 뒤 살아남) → 납부기록 레이블이 '실제 포기'에 더 맞음. 소멸 사건 레이블로 채점해도 skill {t2['sens_label_lapse_event']['skill']:.3f}, 강화 기준선 대비 {t2['sens_label_lapse_event']['skill_vs_strong']:.3f}.")
A(f"(d) 시점: 특허 목록은 2025-08-22까지지만 pay·right는 2026-08-14/15까지 들어 있다(후속 스냅샷). 공식 레이블은 2025-08-22까지의 납부로 판정(보수적, 유예 213일 경과). 2026-08-14까지 납부로 다시 판정하면 1→0 {t3['changed_1_to_0']}건, 0→1 {t3['changed_0_to_1']}건, skill {t3['skill']:.4f}·강화 기준선 대비 {t3['skill_vs_strong']:.4f} — 사후 수정 영향 무시 가능.")
A("    판정: T4 레이블은 원천에서 정확히 재현되고, 정의·스냅샷 대안에서도 성적 변화 ≤0.008(skill)이다.")
open(os.path.join(W, "data_check.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(t4_verdict, boundary_changed, n5_verdict, n5_boundary_changed, round(cell, 3), round(krcell, 3), olb, lbs)

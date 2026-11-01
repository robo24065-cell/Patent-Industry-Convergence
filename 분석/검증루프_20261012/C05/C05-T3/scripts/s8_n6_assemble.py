# -*- coding: utf-8 -*-
"""C05-T3 s8: N6 재채점 결과 모음 -> n6_rescore.json (s4·s5·s5b·s5c·s5d·s6 산출 + 공식 JSON 대조)"""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
J = json.load(open(os.path.join(B, r"모델경쟁_20260927\N6_정부RnD기업_첫특허전환\final_result_r5.json"), encoding='utf-8'))
s4 = json.load(open(os.path.join(CA, 's4_out.json'), encoding='utf-8')); s5 = json.load(open(os.path.join(CA, 's5_out.json'), encoding='utf-8'))
s5b = json.load(open(os.path.join(CA, 's5b_out.json'), encoding='utf-8')); s5c = json.load(open(os.path.join(CA, 's5c_out.json'), encoding='utf-8'))
s5d = json.load(open(os.path.join(CA, 's5d_out.json'), encoding='utf-8')); s6 = json.load(open(os.path.join(CA, 's6_out.json'), encoding='utf-8'))
off = s5['official_from_preds']
cmp = {k: {'rebuild_official_rule': off[k2], 'json': J[k], 'abs_diff': abs(off[k2] - J[k])} for k, k2 in
       [('auc', 'auc'), ('auc_M0', 'auc_M0'), ('skill', 'skill'), ('top1', 'top1'), ('top5', 'top5'), ('top10', 'top10'), ('top1_M0', 'top1_M0'), ('base_rate', 'base'), ('n', 'n'), ('pos', 'pos')]}
me = s5['combos']['ME(F_me,A_me,K_me)']
res = {
 'A_official_rule_rebuild(자기 코드, 트랙 규칙 확인 후)': {
   'units_match': s5b['merge'], 'y_mismatch': s5b['y_mismatch'], 'n': s5b['n'], 'pos': s5b['pos'],
   'metrics_vs_json': cmp, 'max_abs_diff': max(v['abs_diff'] for v in cmp.values()),
   'verdict': '재현(단위 100%, y 100%, 지표 차 0)',
   'note': '첫 시도(s5 OFF 변형)는 단위 2+1·양성 +2 차이 → 원인: 최빈 상호 동점 처리(정렬 vs value_counts 순서) = 정의 해석. value_counts 순서 사용 시 완전 일치(s5b).'},
 'B_literal_definition_rebuild(정의 문장만, 트랙 코드 보기 전)': {
   'n': s4['n_rebuild'], 'pos': s4['pos_rebuild'], 'unit_cmp': s4['unit_cmp'], 'unit_jaccard': s4['unit_jaccard'], 'y_agree_common': s4['y_agree_common'],
   'common_units_rescore': {k: me[k] for k in ['common_auc', 'common_auc_M0', 'common_skill', 'common_top1']},
   'official_units_with_rebuilt_y': {k: me[k] for k in ['offunits_pos', 'offunits_auc', 'offunits_auc_M0', 'offunits_skill', 'offunits_top1', 'offunits_y_agree']},
   'dskill_common_vs_json': me['common_skill'] - J['skill'], 'dskill_offunits_vs_json': me['offunits_skill'] - J['skill'],
   'verdict': '미재현(단위 Jaccard 0.982 < 0.995) — 원인 분류: 정의 해석/원천 버전, 코드 결함 아님, |Δskill| ≤ 0.005',
   'component_attribution': {k: {kk: v[kk] for kk in ['n', 'pos', 'jaccard', 'y_agree_common', 'common_skill', 'offunits_skill']} for k, v in s5['combos'].items()},
   'cause': '[F] 기업→출원인코드 이름 보조 연결의 후보 범위: 공식은 사업자번호를 가진 출원인(r3nts_applicant_status 278,766)에서만 유일명 검사, 자기 판은 CORP_APPLICANT 전체(385,256)·상호 2019–21 최빈. 이름 연결 기업 787 대 480. 불일치 단위 184/37 중 183은 자기 판 이름 연결로 2021년 이전 알려진 출원이 생겨 위험집합에서 빠짐.',
   'name_link_quality_183': {'ca_bizno_blank': s5c['ca_bizno_blank'], 'apps_top10': s5d['apps_top10'], 'firms_le10apps': s5d['firms_le10apps'],
                             'note': '2곳은 출원 3만·2.4만 건 대기업 코드와 동명 매칭(오연결 가능성 큼), 154곳은 출원 10건 이하. 자기 판 이름 매칭의 정밀도가 공식보다 높다고 볼 근거 없음 → 공식 규칙 유지가 합리적, 차이는 한계로만 기록'},
   'A_component(app_firm vs 자기 registry+유일명)': {'pairs': s5['A_pairs_cmp'], 'known_date_diff_rows': s5['known_off_vs_me_diff_rows']}},
 'C_blank_registration_2023_11plus': {k: s6[k] for k in ['kr_rgst_blank_total', 'blank_docs_in_rightlist', 'blank_docs_rgst_year', 'blank_docs_min_rgst_date', 'blank_docs_window_appl',
     'blank_docs_pub_le_2021', 'blank_docs_linked_to_pop_firms', 'pop_firms_with_blank_docs', 'blank_pairs_already_captured_by_app_firm', 'blank_pairs_missed',
     'risk_firms_y_0to1', 'risk_firms_should_drop_known_le2021', 'rescore']},
 'D_KR2024_collection_gap': {'apps_used_after_2023': s6['fa_appl_after_2023'], 'verdict': 'N6 공식 레이블(출원 ≤2023-12-31)·위험집합(알려진 날 ≤2021-12-31)·피처(≤2021)에 2024+ 레코드 0건 → 무관. future 점수(t0=2023, 결과창 2024–25)에는 해당'},
 'E_Dec2023': {'final_window_last_month_share_allfirms': s6['final_window_2022_23_last_month_share_allfirms'], 'dev_window_2020_21_last_month_share_allfirms': s6['dev_window_2020_21_last_month_share_allfirms'],
               'note': 'app_firm 기준 결과창 마지막 달 비중 6.31% 대 개발창 6.18% — 기업 연결 출원에서는 12월 결손 신호가 뚜렷하지 않음(meta 전체 기준 2023-12 비율은 2016–22 평균 대비 약 5.6% 낮음, s3)'},
 'perform_grp_correction_reference': 'C02-T1: 교정 모집단 12,545(+231 중견, −14 정부부처) — 이번 바퀴 재측정 안 함',
}
json.dump(res, open(os.path.join(OUT, 'n6_rescore.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])

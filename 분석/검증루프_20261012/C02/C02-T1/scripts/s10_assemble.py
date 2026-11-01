# 산출 정리: audit_tables.csv(긴 형식 통합), impact.json(판정 포함)
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
J = lambda n: json.load(open(os.path.join(OUT, n), encoding='utf-8'))
parts = []
t = pd.read_csv(os.path.join(OUT, '_tab_a_totals.csv'))
t.loc[t.year == 2006, ['official_exec', 'ratio_gov', 'ratio_gov_plus_secret', 'ratio_gov_vs_off_minus_defense', 'secret_share_funds_vs_off', 'defense_share_off']] = np.nan  # e-나라 2006 셀 파싱 불량(88)
parts.append(t.assign(table='a_totals_official_vs_collected'))
parts.append(pd.read_csv(os.path.join(OUT, '_tab_a_performer.csv')).assign(table='a_performer_share'))
parts.append(pd.read_csv(os.path.join(OUT, '_tab_b_region.csv')).assign(table='b_region_share'))
parts.append(pd.read_csv(os.path.join(OUT, '_tmp_year.csv')).assign(table='b_raw_year_audit'))
parts.append(pd.read_csv(os.path.join(OUT, '_tab_b_pgrp_mislabel.csv')).assign(table='b_perform_grp_mislabel'))
parts.append(pd.read_csv(os.path.join(OUT, '_tab_e_n6_population.csv')).assign(table='e_n6_population_official_vs_corrected'))
parts.append(pd.read_csv(os.path.join(OUT, '_tab_e_n7_secret_scenario.csv')).assign(table='e_n7_secret_route_scenario'))
A = pd.concat(parts, ignore_index=True)
A.to_csv(os.path.join(OUT, 'audit_tables.csv'), index=False, encoding='utf-8-sig')
print(A.table.value_counts())
# 판정
tt = t[t.year.between(2007, 2024)]
P = pd.read_csv(os.path.join(OUT, '_tab_a_performer.csv')); Pq = P[P.year.between(2007, 2024)]
R = pd.read_csv(os.path.join(OUT, '_tab_b_region.csv')); Rq = R[R.year.between(2007, 2024)]
S = pd.read_csv(os.path.join(OUT, 'bias_smd.csv'))
yr = pd.read_csv(os.path.join(OUT, '_tmp_year.csv'))
mis = J('_tab_e_mislabel.json'); rw = J('_tab_e_reweight.json'); lp = J('_tab_e_lapse.json'); lag = J('_tab_d_lag.json'); lk = J('_tab_c_link.json')
n6 = pd.read_csv(os.path.join(OUT, '_tab_e_n6_population.csv'))
o = n6[n6.version == 'official'].set_index('t0'); c = n6[n6.version == 'corrected'].set_index('t0')
imp = dict(
  P1_totals=dict(min_ratio_gov_nonsecret=float(tt.ratio_gov.min()), year_of_min=int(tt.loc[tt.ratio_gov.idxmin(), 'year']), range_ratio_gov=[float(tt.ratio_gov.min()), float(tt.ratio_gov.max())],
                 range_ratio_gov_plus_secret_total=[float(tt.ratio_gov_plus_secret.min()), float(tt.ratio_gov_plus_secret.max())],
                 count_ratio_2018_2024=[float(tt.ratio_n.min()), float(tt.ratio_n.max())], years_ratio_lt_0_80=tt[tt.ratio_gov < 0.80].year.tolist(),
                 verdict='통과(경계): 비보안 정부연구비/공식 집행액 0.802–0.880, 0.80 미만 연도 없음(2024 0.802). 과제 수 2018–2024 비율 1.000. 누락분은 보안과제 정부연구비 비공개: 보안과제 총연구비를 더하면 0.985–1.005, 공식에서 국방 목적 집행액을 빼고 비교하면 0.968–1.012.'),
  P2_composition=dict(max_abs_diff_pp=float(Pq.diff_pp_nonsecret.abs().max()), performers_years_gt5pp=Pq[Pq.diff_pp_nonsecret.abs() > 5][['year', 'performer', 'diff_pp_nonsecret']].round(2).to_dict('records'),
                      region_max_abs_diff_pp=float(Rq.diff_pp.abs().max()), region_years_gt5pp=int((Rq.diff_pp.abs() > 5).sum()),
                      ministry='판정 불가(공식 부처별 표 미확보; e-나라 133004는 경제사회목적별). 보안과제 총연구비 / 국방 목적 공식 집행액 0.93–1.18(총연구비라 민간 부담 포함). 수집본 비보안 국방부 과제 7건, 방위사업청 0건.',
                      verdict='단서 필요: 출연연 −5.1~−8.0%p(2020–2024), 대학 +5.05%p(2020)·+5.20%p(2024), 대전 −5.2~−8.5%p(2010–2024 다수).'),
  P3_official_reliability=dict(amount_2018_2023='e-나라 1330 = KIST 융합연구 성과분석 2022·2023 인용 조사·분석 값 일치(6개 연도 차이 0)', amount_2024='e-나라 262,175억 = 인사이트 보도 약 26.2조(일치)',
                               count='2018–2022 KIST2022 표, 2019 뉴시스(구간별 합 70,327), 2020 뉴스토마토 73,501, 2023 KIST2023, 2024 인사이트 60,696 — 수준 일치',
                               uncertain='2024 보도 전년 대비 증감(집행 −9.6%, 과제 −13%)은 1330·KIST 2023 값으로 계산한 −14.2%, −15.5%와 다름(2024 보고서의 2023 기준값 재작성 가능성) → 2023→24 증감률은 불확실. 2007–2017 공식 과제 수 미확보(판정 불가). e-나라 2006 셀 파싱 불량으로 제외.'),
  P4_raw=dict(dup_pjt_year=0, gov_na_nonsecret_max_2007_2024=float((yr[yr.year.between(2007, 2024)].gov_na - yr[yr.year.between(2007, 2024)].secret).max()),
              gov_neg=0, secret_share_range=[float(yr[yr.year.between(2007, 2024)].secret.min()), float(yr[yr.year.between(2007, 2024)].secret.max())],
              firm_bizno_missing_max_2010plus=float(yr[yr.year >= 2010].firm_bizno_missing.max()), ministry_family_jumps_2007plus='OTHER 2021 +38%(다부처 +0.43조·질병관리청 신설; 실제 변화로 판단)',
              secret_rows=mis['secret_rows'], secret_masked=mis['secret_masked_share'],
              perform_grp_bug=dict(desc='ntis_B_build_raw.py GRP 매핑 오류: 코드 06(정부부처)→"중견기업", 코드 08(중견기업)→"기타"', rows_06=1973, rows_08=10456,
                                   n9_2022=mis['n9_2022']),
              verdict='경고: perform_grp 오표기(데이터 오류). 그 외 중복·결측·음수·사업자번호 기준 통과.'),
  P5_link_smd=dict(flags=S[S['flag_abs_gt_0.25']][['block', 'variable', 'smd']].round(3).to_dict('records'), N6_panel_linked_share=lk['N6_panel_linked'],
                   N6_unlinked_ambiguous_name_share=lk['N6_unlinked_with_ambiguous_name_match_share'], N7_root_linked_share=lk['C_N7_root_linked_share'], N7_root_ksic5_TV=lk['C_ksic5_TV'],
                   verdict='단서 필요: 연결 과제는 연구비가 크고(SMD 0.94) 산업부 계열(0.51)·출연연·개발연구 쪽, 교육부·국공립연·기타 부처는 적다. N6 연결 기업은 법인(1.21)·과제 수(0.40)·연구비(0.51)가 크다. N7 모집단 근원에서 사업자번호 미연결 법인은 과거 출원 법인(최근 출원연도 SMD 1.57)로 생존 편향. KSIC 구성 차이는 작음(2자리 |SMD|≤0.11, 5자리 TV 0.12).'),
  P6_reporting_lag=dict(curve=dict(link_rate_rel_2019_21={'2022': 1.199, '2023': 1.184, '2024': 1.173, '2025': 0.262}, ntis_count_rel={'2022': 1.201, '2023': 1.178, '2024': 1.138, '2025': 0.180},
                                   official_2022_cmp=lag['official_cmp']),
                        N9=dict(test_window='출원 2022–23(스냅샷까지 3.3–4.3년)', est_unreported_share='0–5%(2.25년 경과 2024 출원도 연결률·건수가 2019–21 이상; 결함 신호 없음)', verdict='통과(10% 미만)',
                                delayed_version='정당: 출원 후 1.25년(2025 출원) 완결도 0.18–0.26 → t−1 출원 성과는 학습 시점에 대부분 미보고(90% 미만).',
                                caveat='2022–24 출원 연결률이 2019–21보다 15–20% 높다(제도·보고 관행 변화 가능). 시험 기저율 6.97%가 개발 6.1–6.8%보다 높은 배경일 수 있음.'),
                        N7=dict(test_window='과제연도 2023–24', project_registration_completeness='2023 71,799/71,804, 2024 60,693/60,696(99.99%)', verdict='통과: 과제 등록 지연 없음. "2024 등록 지연 −15%" 서술은 틀림(공식도 −15.5%; 예산 삭감에 따른 실제 감소).',
                                secret_route=rw['N7']),
                        N6=dict(verdict='해당 없음(결과=제공 KR 출원). 모집단 과제 기록 완결.')),
  P7_leak=dict(verdict='통과: N6(RP/LP 성과 Year ≤ t0−1), N7(f_M2·gov_rg Year ≤ t0, first_ntis ≤ t0 제외), N9(이력 출원 ≤ v−2, 결과는 정의상 NTIS 연결) — 결과 구간 특허에 NTIS 플래그를 피처로 쓴 곳 없음. 단 Year ≤ t0 성과도 스냅샷이라 t0 이후 보고분이 섞이는 준-정합(기존 한계와 같음).'),
  P8_reweight=dict(N9=rw['N9'], N6=rw['N6'], N7=rw['N7'], N6_mislabel=dict(final_t0_2021_official=int(o.loc[2021, 'n']), final_t0_2021_corrected=int(c.loc[2021, 'n']),
                   midsize_only_excluded_final=int(c.loc[2021, 'n_only_midsize(08)']), govdept_only_included_final=int(o.loc[2021, 'n_only_govdept(06)']),
                   dev_rate_change_pp_max=float(100 * (c.loc[2010:2019, 'rate'] - o.loc[2010:2019, 'rate']).abs().max())),
                   verdict='통과: 재가중 양성률 변화 N9 +3.1%, N6 −0.4%, N7 누락 양성 추정 4.9%(시나리오 2.8–7.4%, 극단 상한 23.5%); KSIC Spearman 0.98–1.00. N7 개별 KSIC는 최대 +29%(대·중견 첫 진입 근사 배분).'),
  P9_lapse=dict(recompute=lp['recompute'], by_applicant_type=lp['by_applicant_type'], gov_comp=lp['gov_composition'], nongov_comp=lp['nongov_composition'], standardized=lp['standardized'],
                defense=lp['defense_sensitivity'],
                verdict='숫자 ok(14.23%·20.97%). 사전 기준(비기원 구성 재가중·등록연도·KSIC 표준화) 격차 3.49–3.56%p ≥3%p, 원격차 6.74%p의 52% → 형식상 "유지"(경계). 정부기원 구성 기준 표준화는 1.36–1.64%p(절반 이하). 대학 특허끼리는 12.1% 대 11.8%(차이 없음), 출연연 15.0% 대 15.6%, 기업 16.6% 대 20.6%. 격차 대부분은 대학·출연연 출원인 구성 → 서술 단서 필요.'),
)
json.dump(imp, open(os.path.join(OUT, 'impact.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('ok')

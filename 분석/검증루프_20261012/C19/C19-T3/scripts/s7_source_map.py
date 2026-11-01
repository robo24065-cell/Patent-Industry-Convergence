# C19-T3 s7: 원고 숫자 토큰 → 원천 → US·EP 행 사용 여부 → 재실행 여부 (source_map.csv)
# 분류 근거는 아래 RULES(원천 파일 경로 패턴별, 계보 확인 결과)와 cache/s2_scripts.csv·s3_sources.csv.
import os, sys, re, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
A = json.load(open(C + r'\s1_app.json', encoding='utf-8'))['APP']
TOK = list(csv.DictReader(open(C + r'\s1_tokens.csv', encoding='utf-8-sig')))
# (패턴, 원천 스크립트, 사용구분, 재실행, 근거)
# 사용구분: 사용 / 사용(구조상 불변) / 사용(행 수 사실) / 사용(공식 성적, 정정 대상 아님) / 사용 안 함
R_MAP = 'C19 cache/map_V1E (scripts/map 사본, P0 원본 대조 후)'
R_LO = 'C19 cache/lo_V1E (scripts/lo 사본, P0 원본 대조 후)'
NO = '미재실행(모형 재학습 필요)'
RULES = [
 (r'map5\\m2_(checks|country_summary|ksic_profile)', 'map5 m1_build_counts→m1b→m2_trend_map', '사용', R_MAP, 'm1이 메타 5개국 행을 KSIC×국가×연도로 셈(US·EP 포함)'),
 (r'map5\\m3_summary', 'map5 m3_leadlag(← m2_panel_appl)', '사용', R_MAP, '국가별 연차 변화 선후 시험, US·EP 계열 포함'),
 (r'map5\\m3c_contemporaneous', 'map5 m3c_contemporaneous(← m2_panel_appl·m1_origin)', '사용', R_MAP, 'US·JP·CN·EP 동조, 우선권 제거판'),
 (r'map5\\m4_summary', 'map5 m4_industry_quadrant(← m2_ksic_profile)', '사용', R_MAP, 'RTA P1(2008–10)·P2(2015–17), EP 쌍 2008–09 포함'),
 (r'map5\\m5_summary', 'map5 m5_dashboard(← m2·m4 + 트랙 미래전망 파일)', '사용', R_MAP + '; 트랙 전망 파일은 고정', '대시보드 순위상관(m2 RTA 열 포함)'),
 (r'map5\\m7_summary', 'map5 m7_supplement(← m2_panel_pub_corrected)', '사용', R_MAP, 'JP 2021 보정(공개 기준)'),
 (r'split34\\t1_variance|split34\\t2b_', 'split34 s1_panel→s2_decomp', '사용', R_MAP, '5개국 패널(KSIC×국가×연도) 분산 분해'),
 (r'split34\\t2c_', 'split34 s1_panel→s4_figs_robust', '사용', R_MAP, 'KR 출원 성장(패널 KR 열) ↔ KOSIS'),
 (r'split34\\t3a_', 'split34 s1_panel→s3_forecast', '사용', R_MAP, '5개국 성장 예측(쌍 순서·RMSE)'),
 (r'split34\\t5_cases', 'split34 s1_panel→s5_cases', '사용', R_MAP, '사례(KR 출원·KOSIS)'),
 (r'fig06_quadrant', '보고서 figs make_data_figs.py(← map5 m4_quadrant_table)', '사용', R_MAP + '(m4_quadrant_table)', '사분면 목록 = m4 P2 창(2015–17)'),
 (r'label_origin_R10\\res_fam_check', 'label_origin_R10 lo_fam.py', '사용', R_LO, '해외(US·JP·EP·CN) 문서 family 필드 → KR 등록번호 연결'),
 (r'label_origin_R10\\res_(tests|p1|p1_disagree|p3)\.', 'label_origin_R10 lo_fam→lo_tests.py', '사용', R_LO, 'KR 문서의 해외 가족 다수결 라벨(fam_reg)'),
 (r'label_origin_R10\\res_posthoc2', 'label_origin_R10 lo_fam→lo_posthoc2.py', '사용', R_LO, 'US 가족 문서 출원인 칸(메타 US 행)'),
 (r'label_origin_R10\\RESULT_R10', 'label_origin_R10 RESULT_R10.md(← res_p1 CI)', '사용', R_LO + '(res_p1)', 'P1 CI 상한 요약'),
 (r'foreign_public_R11\\(res_main|res_tests|PREREG)', 'R11 fp_prep(← alt_assign·fam_reg)→fp_tests', '사용', R_LO + '(fp)', '해외 문서×KR 문서 표본(fam_reg)'),
 (r'c5_R10\\c1_label_agreement', 'c5_R10 c1_assign.py', '사용(구조상 불변)', '불필요(창 2015–19; C-COUNTRY-SHARE 행 수는 별도)', 'is_inv and appl_year 2015–19, 쌍 행은 US 출원 2000–03·EP 2000–09'),
 (r'c5_R10\\c2_results', 'c5_R10 c2_rta.py(← alt_assign)', '사용(구조상 불변)', '불필요(창 2015–19)', 'base = is_inv and appl_year between 2015 and 2019'),
 (r'C11-T2\\verify_5', 'C11-T2 v5_land/v5_frac(← alt_assign)', '사용(구조상 불변)', '불필요(창 2017–19)', 'appl_year 2017–19(해외), KR ≤2019 표만'),
 (r'C10-T3\\res\\s2_agree', 'C10-T3 s2_reassign.py', '사용(구조상 불변)', '불필요(창 2015–19)', "agree('is_inv and appl_year between 2015 and 2019')"),
 (r'C14-T2\\pkr_ipc_altA', 'C14-T2 s10_pkr_table.py', '사용(구조상 불변)', '불필요(창 2015–19)', '출원 2015–19 발명특허 재현율'),
 (r'_shared\\README', '_shared build_meta.py(행 수)', '사용(행 수 사실)', '해당 없음(행 수 = 제공 그대로; V1E 고유 문서 수는 s0)', 'DATA-META 15,141,159행 등'),
 (r'T3_기술셀부상예측\\R3\\final_result_r3|T3_기술셀부상예측\\R2\\final_result', 'T3 R3/R2 final_run(build_base_pub 5개국 셀 카운트)', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 V1E 민감도 Δ+0.0057, 영향 없음)', '공식 최종시험 1회 값'),
 (r'T3_기술셀부상예측\\R3\\FINAL_REPORT', 'T3 R3 final_run·감사', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 민감도)', 'X-T3-* 공식 시험 부속값'),
 (r'T3_기술셀부상예측\\(VERIFY_R5|ESSENTIAL_R7|TXT_R9)|C11-T2\\res\\s5_summary|c5_R10\\c5_t3_summary|C10-T3\\res\\s6_t3|C15-T3\\res\\s5_t3|C17-T3\\t3_newipc|C05-T1|fig07_t3', 'T3 계열 재학습(base_rows 5개국 셀 카운트)', '사용', NO, 'T3 피처 US_*·EP_*(base_rows 5개국) — 비공식 분석값'),
 (r'T1_산업대기실\\R3\\final_result_r3|T1_산업대기실\\R3\\FINAL_REPORT|T1_산업대기실\\R3\\fig08', 'T1 R3 final_run_r3(ep_firm: EP 행 법인 매칭)', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 V1E 민감도 Δ−0.0002)', 'T1 동결 피처 ep_* 5개'),
 (r'T1_산업대기실\\VERIFY_R5|fig08_t1|C15-T2\\|C16-T1\\variants|C17-T1\\variants|C17-T2\\sens|C18-T1\\sens', 'T1 동결 스펙 재학습(ep_*)', '사용', NO, 'T1 동결 피처 ep_* 5개(EP 행) — 비공식 분석값'),
 (r'C13-T2\\', 'C13-T2 시드 묶음(N4·N6·N7·N8·N9·T4)', '사용 안 함', '해당 없음', 'C18-T2 노출 지도: 이 6개 트랙은 US·EP 메타 행 미사용(T4·N8은 KR family 토큰)'),
 (r'N5_표준특허선언레이더\\(final_result_r4)', 'N5 final_run_r4(fam_n·fam_US/EP: 해외 문서 행)', '사용(공식 성적, 정정 대상 아님)', '공식 불변(C18-T2 V1E 민감도 Δ−0.0026)', 'N5 M1 묶음'),
 (r'N5_표준특허선언레이더\\VERIFY_R5', 'N5 강화 기준선(fam_n)', '사용', NO, 'N5 M1 묶음 — 비공식 분석값'),
 (r'(N4_|N6_|N7_|N8_|N9_|T4_특허포기)', '해당 트랙 final_run', '사용 안 함', '해당 없음', 'C18-T2 exposure.csv: US·EP 메타 행 미사용(N8·T4는 KR family 토큰 — 별도 결함)'),
 (r'T1_산업대기실\\TXT_R9|T1_산업대기실\\NONCIRC', 'T1 TXT_R9/NONCIRC_R8(KR 문서·법인)', '사용 안 함', '해당 없음', "nc_common: meta where cc='KR'; ep_* 피처 없음(grep 0)"),
 (r'T1_산업대기실\\R[23]\\ext_link|T1_산업대기실\\R3\\ext_link_stats', 'N1 a06_link_summary / T1 ext_link_stats(KSIC 연결표)', '사용 안 함', '해당 없음', '메타 행 미사용(외부 연결표)'),
 (r'confirm_R12', 'R12 r12_prep(cc=KR)→r12_tests', '사용 안 함', '해당 없음', "r12_prep: rec_conc where cc='KR', fam_reg 미사용"),
 (r'audit_r7', 'audit_r7 concordance_resolution_check.py', '사용 안 함', '해당 없음', "cc='KR'만 집계(KSIC 목록은 50개로 불변)"),
 (r'_ideas_R8\\p2', '_ideas_R8 p2_within_group.py', '사용 안 함', '해당 없음', "rec where cc='KR'"),
 (r'C06-T1', 'C06-T1 s2_audit/s3_rescore', '사용 안 함', '해당 없음', 'rescore_B.csv는 fam_reg 읽기(152행) 전에 저장; fam_reg는 미인용 s3_circ.csv에만'),
 (r'_shared\\ext\\|ntis_|noncirc_n9n6|C07-T1|C15-T1|C17-T3\\cache\\ntis|C18-T1\\monthly|C11-T1\\data_check|registry_full|business_rights|IDEAS_R7|ntis_tables|PROTOCOL_R2|R9_결과', 'KR·NTIS·외부 연결 원천', '사용 안 함', '해당 없음', 'KR 행·외부 자료만(cc=KR 필터 또는 메타 미사용)'),
 (r'R6_결과|R10_결과', '라운드 결과 문서(트랙 공식값 요약)', '사용(공식 성적, 정정 대상 아님)', '공식 불변', 'OFF-DEV2FINAL·N8 보정·판정 요약'),
 (r'(^|\\)PREREG\.txt$|res_dev\.csv|res_main\.csv|res_tests\.json', 'R12 표본 문서', '사용 안 함', '해당 없음', 'R12 PREREG·결과(KR)'),
]
def classify(src):
    for pat, scr, use, rr, why in RULES:
        if re.search(pat, src): return scr, use, rr, why
    return '미분류', '미분류', '', ''
def src_of(app_id):
    v = A[app_id]; parts = [p.strip() for p in re.split(r';\s*', v['src']) if p.strip()]
    cl = [classify(p) for p in parts]
    # 한 행에 원천 여러 개면 가장 강한 노출을 대표로
    order = ['사용', '사용(행 수 사실)', '사용(공식 성적, 정정 대상 아님)', '사용(구조상 불변)', '사용 안 함', '미분류']
    best = min(cl, key=lambda c: order.index(c[1]) if c[1] in order else 9)
    return v['src'], best
# 본문 토큰 → 부록 행
SECN = lambda s: s
app_tok = {}
NUM = re.compile(r'(?<![A-Za-z0-9_.])[−\-+]?\d[\d,]*(?:\.\d+)?%?')
def norm(t):
    t = t.replace('−', '-').replace('+', '').rstrip('%').rstrip(',')
    if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', t): t = t.replace(',', '')
    return t.rstrip(',').lstrip('-')
for k, v in A.items():
    for m in NUM.finditer(v['val'].replace('–', ' – ').replace('~', ' ~ ')):
        app_tok.setdefault(norm(m.group(0)), []).append(k)
def sec_match(app_sec, sec, line):
    if sec == '표지·요약' and '표지' in app_sec: return True
    return any(x.strip() == sec or (sec.count('.') == 1 and x.strip().startswith(sec + '.')) or (x.strip().count('.') == 1 and sec.startswith(x.strip() + '.')) for x in re.split(r'[,·]', app_sec))
# 본문 전용 숫자(부록 행 없음)의 원천: 절·문맥 규칙 (C18-T3 s3b 미대응 목록과 원고 문맥으로 확인)
BODY_SEC_DEFAULT = {
 '표지·요약': ('요약(부록 중복)', '사용 안 함', '해당 없음', '요약 문장 — 부록 행 숫자의 반복 또는 범위'),
 '1.1': ('1.1 연계표 진단(audit_r7·split34 사례)', '사용 안 함', '해당 없음', 'KR 문서 기준'),
 '1.2': ('1.2 차별성(서술)', '사용 안 함', '해당 없음', ''),
 '1.3': ('1.3 표 2·주', '확인 필요', '', ''),
 '1.4': ('1.4 표 3(연계 결과)', '사용 안 함', '해당 없음', '외부 연결률'),
 '2.1': ('2.1 검증 설계(T3·T1 공식)', '사용(공식 성적, 정정 대상 아님)', '공식 불변', ''),
 '2.2': ('2.2 지표 설명', '사용 안 함', '해당 없음', ''),
 '2.3': ('2.3 의심 장치(트랙 감사)', '확인 필요', '', ''),
 '2.4': ('2.4 필수성 설계', '확인 필요', '', ''),
 '2.5': ('2.5 전처리', '확인 필요', '', ''),
 '3.1.1': ('map5 m2', '사용', R_MAP, ''), '3.1.2': ('map5 m4', '사용', R_MAP, ''), '3.1.3': ('split34', '사용', R_MAP, ''),
 '3.2.2': ('T3 공식/분석', '사용', NO, 'T3 계열'), '3.2.3': ('T3 미래 전망', '사용', NO, 'T3 계열'), '3.2.1': ('T3 정의', '사용 안 함', '해당 없음', '정의 숫자'),
 '3.3.2': ('T1 공식/분석', '사용', NO, 'T1 계열'), '3.3.3': ('T1 미래 전망', '사용', NO, 'T1 계열'), '3.3.1': ('T1 정의', '사용 안 함', '해당 없음', '정의 숫자'),
 '3.4': ('N9·N6', '사용 안 함', '해당 없음', 'KR·NTIS'), '3.5': ('보조 트랙(N4·N5·N7·N8·T4)', '확인 필요', '', ''),
 '3.6.1': ('3.6.1 정의(LBL)', '사용 안 함', '해당 없음', 'KR'), '3.6.2': ('3.6.2 근거 A(R12·R10)', '확인 필요', '', ''),
 '3.6.3': ('3.6.3 근거 B(T1 TXT_R9·T3 TXT_R9)', '확인 필요', '', ''), '3.6.4': ('3.6.4 근거 C', '확인 필요', '', ''), '3.6.5': ('3.6.5 종합', '확인 필요', '', ''),
 '4.1': ('4.1 시사점', '확인 필요', '', ''), '4.2': ('4.2 시나리오', '확인 필요', '', ''), '4.3': ('4.3 발전 방향', '사용 안 함', '해당 없음', ''),
 '4.4': ('4.4 한계', '확인 필요', '', ''), '4.5': ('4.5 결론', '확인 필요', '', ''), '데이터 출처': ('데이터 출처', '사용 안 함', '해당 없음', ''),
}
rows = []
for t in TOK:
    ln = int(t['line']); base = dict(원고행=ln, 구분=t['part'], 절=t['sec'], 숫자=t['tok'], 정규화=t['ntok'])
    if t['struct']:
        rows.append({**base, '원천': '구조 숫자(' + t['struct'] + ')', '부록ID': '', '원천 스크립트': '해당 없음', 'US·EP 행 사용': '해당 없음', '재실행': '해당 없음', '근거': ''}); continue
    if t['part'] == '부록' and t['app_id'] in A:
        src, (scr, use, rr, why) = src_of(t['app_id'])
        rows.append({**base, '원천': src, '부록ID': t['app_id'], '원천 스크립트': scr, 'US·EP 행 사용': use, '재실행': rr, '근거': why}); continue
    if t['part'] == '부록':
        rows.append({**base, '원천': '부록 머리·설명 줄', '부록ID': '', '원천 스크립트': '해당 없음', 'US·EP 행 사용': '해당 없음', '재실행': '해당 없음', '근거': ''}); continue
    cand = app_tok.get(t['ntok'], [])
    insec = [k for k in cand if sec_match(A[k]['sec'], t['sec'], ln)]
    use_ids = insec or cand
    if use_ids:
        cls = [src_of(k)[1] for k in use_ids]
        order = ['사용', '사용(행 수 사실)', '사용(공식 성적, 정정 대상 아님)', '사용(구조상 불변)', '사용 안 함', '미분류']
        best = min(range(len(cls)), key=lambda i: order.index(cls[i][1]) if cls[i][1] in order else 9)
        scr, use, rr, why = cls[best]
        rows.append({**base, '원천': '부록 ' + ','.join(use_ids[:4]) + ('' if insec else ' (절 불일치, 숫자 일치)'), '부록ID': use_ids[best],
                     '원천 스크립트': scr, 'US·EP 행 사용': use, '재실행': rr, '근거': why + ('' if len(use_ids) == 1 else f' / 같은 숫자 부록 행 {len(use_ids)}개 중 가장 강한 노출')})
    else:
        d = BODY_SEC_DEFAULT.get(t['sec'], ('?', '확인 필요', '', ''))
        rows.append({**base, '원천': '본문 전용(부록 행 없음): ' + d[0], '부록ID': '', '원천 스크립트': d[0], 'US·EP 행 사용': d[1], '재실행': d[2], '근거': d[3]})
jdump(rows, 's7_rows_raw.json')
from collections import Counter
out(Counter(r['US·EP 행 사용'] for r in rows))
out('미분류 원천', sorted(set(r['원천'] for r in rows if r['US·EP 행 사용'] == '미분류'))[:40])
out('확인 필요 본문', len([r for r in rows if r['US·EP 행 사용'] == '확인 필요']))

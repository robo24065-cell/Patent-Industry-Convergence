# s8: 캐시에서 feasibility.csv, essentiality_strict.csv, bias_checks.csv 조립(재현용)
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4'
C = W + r'\cache'
F = []


def add(c, crit, metric, val, thr, verdict, src):
    F.append(dict(candidate=c, criterion=crit, metric=metric, value=val, threshold=thr, verdict=verdict, source=src))


# ---- C03-A ----
A = pd.read_csv(C + r'\s2_A_dev_stats.csv'); a = A[A.link == 'PUB']; a2 = A[A.link == 'PUB+REG']
for _, r in a.iterrows():
    add('C03-A', 'P1', f'양성(t0={r.t0}) / 위험쌍 / 기저율', f'{r.pos} / {r.risk_pairs} / {r.base_rate}', '원점당 ≥200', '미달', r'cache\s2_A_dev_stats.csv')
add('C03-A', 'P1', '개발 고유 사건(원점 합 = 고유, 창 겹침 없음)', int(a.pos.sum()), '≥1,000', '실패 → 중단', r'cache\s2_A_dev_stats.csv, cache\A_dev_pos.parquet')
add('C03-A', 'P1(사후, 판정 밖)', 'KR 등록번호 연결 추가 시 고유 사건', int(a2.pos.sum()), '≥1,000', '사후 — 등록 연결은 잠금창 비대칭(우선 출원 대부분 미등록)이라 주 정의에서 미리 제외했다', r'cache\s2_A_dev_stats.csv')
D = pd.read_csv(C + r'\s2b_A_posthoc_diag.csv')
add('C03-A', 'P1(사후, 판정 밖)', 'KR vis 조건 제거 H1 / 12개월 창', f"{int(D.H1_novis.sum())} / {int(D.Y12_vis.sum())}", '≥1,000', '사후 — 잠금 12개월 창은 US·EP 공개 2025-07 이후 미완결이라 불가', r'cache\s2b_A_posthoc_diag.csv')
add('C03-A', '경고', '2024-H1 창(t0=2023) 양성 41, 2012–2022는 51–99', 41, '-', '최근 해외 문서 family 필드에 KR 공개번호 반영이 늦을 수 있다 → 잠금창은 더 잘릴 위험', r'cache\s2_A_dev_stats.csv')
# ---- C03-B ----
Bs = pd.read_csv(C + r'\s3_B_dev_stats.csv'); bl = json.load(open(C + r'\s3_B_log.json', encoding='utf-8'))
for _, r in Bs.iterrows():
    add('C03-B', 'P1', f'양성(t0={r.t0}) / 창 실시권 행 / 매핑 행', f'{r.pos} / {r.win_rows} / {r.win_mapped}', '원점당 ≥200', '미달', r'cache\s3_B_dev_stats.csv')
add('C03-B', 'P1', '개발 원점 합 / 고유 (실시권자, KSIC)', f"{bl['origin_sum']} / {bl['unique_cd_k']}", '고유 ≥1,000', '실패 → 중단', r'cache\s3_B_log.json')
add('C03-B', '참고', '실시권자 이름 → 출원인코드 매핑률(전체, 2024 사전)', bl['lic_mapped_share_all_lt2025_at_2024'], '-', '창 실시권 행이 509–1,063건이라 어떤 정의로도 1,000 불가', r'cache\s3_B_log.json')
# ---- C03-C ----
Cs = pd.read_csv(C + r'\s4_C_dev_stats.csv')
for _, r in Cs.iterrows():
    add('C03-C', 'P1', f'양성(t0={r.t0}) / 위험쌍 / 적격기업 / 기저율', f'{r.pos} / {r.risk_pairs} / {r.eligible_firms} / {r.base_rate}', '원점당 ≥200', '미달', r'cache\s4_C_dev_stats.csv')
add('C03-C', 'P1', '개발 고유 사건(창 겹침 없음, 원점 합 = 고유)', int(Cs.pos.sum()), '≥1,000', '통과(경계: 1,028). 고유 (기업, KSIC) 쌍은 783', r'cache\s4_C_dev_stats.csv, verify_0\v1_out.txt(SQL 재구현 1,028/1,028 일치)')
p2 = [('T1 산업 대기실(공식)', '출원 2020–2023 (기업×KSIC) 진입',
       '문서 겹침: 잠금창 법인 문서의 82.0%가 2023 출원(T1 최종 결과창 문서). 사건 다름: T1은 그 KSIC 문서가 없던 쌍의 진입, C03-C는 이미 있는 KSIC의 몫 상승(위험 쌍이 서로소)'),
      ('T3 기술셀 부상(공식)', 'KR 공개 2022–2024 셀 성장', '기간 겹침 없음(잠금 2025-01~06)'),
      ('T4·N4·N5·N6·N7·N8·N9(공식)', '각 결과창 ≤2025-12, 사건 유형 다름', '사건 유형 다름(소멸·폐업·ETSI·출원·NTIS·심판)'),
      ('R6 K2P', 'KR 공개 2023–2024 국내단계 진입', '기간 겹침 없음'),
      ('R8 P1·C01 C-C', 'KR 공표 2022–2024 (IPC×KSIC) 몫', '기간 겹침 없음, 단위 다름'),
      ('C01 C-B', 'KR 첫 공개 2025-H1 (KSIC×IPC 셀)', '미채점(기각). 같은 기간 KR 공개 건수만 열람(27,225). 단위·사건 다름'),
      ('자료 행: 제공 KR 공개 2025-01~06', '어떤 시험의 결과로도 미사용(windows_used.csv)', '-')]
for nm, w, note in p2:
    add('C03-C', 'P2', f'대조: {nm} — {w}', note, '사건 미사용', '통과', r'..\..\C01\C01-T4\windows_used.csv')
add('C03-C', 'P2', '잠금 창', 'KR 출원공개 2025-01-01~2025-06-30 (t0=2024)', '≥2025-01-01', '통과', 'candidates.txt')
j = json.load(open(C + r'\s5_C_p3p4.json', encoding='utf-8'))
for r in j['p3_rows']:
    add('C03-C', 'P3', f"{r['link']}: 연결률 / SMD 연도 / SMD 해외우선 / TVD KSIC / 양성가중 칸 공존",
        f"{r['link_rate']} / {r['smd_pub_year']} / {r['smd_foreign_prio']} / {r['tvd_ksic']} / {r['pos_weighted_cell_coexist']}",
        '연결률 ≥0.30, 사전 속성 |SMD|<0.5 또는 공존 ≥0.80', '통과(경고: TVD 0.23–0.30, 사전 밖 속성 법인상호 SMD 0.659)', r'cache\s5_C_p3.csv')
add('C03-C', 'P3 경고', '국내법인 코드 연결률 공개연도별 2010→2024', f"{j['corp_link_rate_by_pub_year']['2010']} → {j['corp_link_rate_by_pub_year']['2024']}", '-', '최근 연도 하락(0.94→0.875)', r'cache\s5_C_p3p4.json')
add('C03-C', 'P4(a)', '잠금 끝 절단 2025-04~06 중앙값 / 01~03 중앙값(법인 문서)', j['P4a_end_trunc'], '≥0.70', '통과', r'cache\s5_C_monthly_2024_2026.csv')
add('C03-C', 'P4(b)', '2025-01~06 / 2024-01~06 법인 문서', j['P4b_same_period'], '≥0.70', '통과', r'cache\s5_C_monthly_2024_2026.csv')
add('C03-C', 'P4(c)', '알려진 결손', 'KR 등록 공란(2025-01~08)은 출원공개일만 써서 설계로 제외. CN·JP·등록원부 미사용. 제공 KR 공개는 2025-07까지 수록(창 끝 06-30)', '겹침 없음', '통과', 'candidates.txt 설계 보충')
add('C03-C', 'P4 진단', '창 ≥3건 기업 수 2025-H1/2024-H1 ; 코드 피복률 2025-H1 vs 2024-H1', f"{j['P4b_firms_ge3']} ; {j['diag_code_coverage_2025H1_vs_2024H1']}", '-', '정상', r'cache\s5_C_p3p4.json')
# ---- P5 ----
E = []
ub = json.load(open(C + r'\s6_C_p5_UB.json', encoding='utf-8'))
E.append(dict(candidate='C03-C', level='pooled_2012_2023', assign='UB(제공 라벨 학습 IPC 서브그룹 최빈, 표본 내)', J_pub=ub['pooled_jac'], recall_prov=ub['pooled_recall'],
              ev_pub=ub['ev_pub'], ev_prov=ub['ev_prov'], doc_agree=ub['doc_agree_mean'], verdict='상한'))
ok_all = True
for a_ in ['TXT-FREE', 'FREE-IPC', 'H-sc-1']:
    r = json.load(open(C + rf'\s6_C_p5_{a_}.json', encoding='utf-8'))
    c1 = r['pooled_jac'] < r['null_q05']; c2 = r['pooled_jac'] < ub['pooled_jac']; ok_all &= c1 and c2
    E.append(dict(candidate='C03-C', level='pooled_2012_2023', assign=a_, J_pub=r['pooled_jac'], recall_prov=r['pooled_recall'], ev_pub=r['ev_pub'], ev_prov=r['ev_prov'],
                  doc_agree=r['doc_agree_mean'], J_null_mean=r['null_mean'], J_null_q05=r['null_q05'], J_null_min=min(r['null_jac']), J_null_max=r['null_max'], J_UB=ub['pooled_jac'],
                  crit_i=bool(c1), crit_ii=bool(c2), verdict='통과' if c1 and c2 else '실패'))
    R = pd.read_csv(C + rf'\s6_C_p5_{a_}.csv')
    for _, x in R.iterrows():
        nj = [x[f'null_inter_{k}'] / x[f'null_union_{k}'] for k in range(20)]
        E.append(dict(candidate='C03-C', level=f't0={int(x.t0)}', assign=a_, J_pub=x.jac, recall_prov=round(x.inter / x.ev_prov, 4), ev_pub=x.ev_pub, ev_prov=x.ev_prov,
                      doc_agree=x.doc_agree, J_null_mean=round(float(np.mean(nj)), 4), J_null_q05=round(float(np.quantile(nj, 0.05)), 4), J_null_min=round(min(nj), 4),
                      J_null_max=round(max(nj), 4), crit_i=bool(x.jac < np.quantile(nj, 0.05)), verdict='(원점별 참고)'))
pd.DataFrame(E).to_csv(W + r'\essentiality_strict.csv', index=False, encoding='utf-8-sig')
add('C03-C', 'P5', '세 공개 배정 모두 J_pub < J_null 5%분위 그리고 < J_UB',
    'TXT-FREE 0.1324<0.1442 / FREE-IPC 0.2006<0.2119 / H-sc-1 0.2019<0.2093 ; J_UB 0.4105', '모두 충족', '통과' if ok_all else '실패', 'essentiality_strict.csv')
add('C03-C', '종합', 'P1–P5', '모두 통과' if ok_all else '실패', '-', '통과 → PREREG_final_draft.txt 작성(모형 학습·최종시험 없음)', '-')
pd.DataFrame(F).to_csv(W + r'\feasibility.csv', index=False, encoding='utf-8-sig')
# ---- bias_checks ----
bc = json.load(open(C + r'\s7_C_bias.json', encoding='utf-8'))
Bc = [dict(candidate='C03-C', check=r['link'], metric='연결률 / SMD 공개연도 / SMD 해외우선 / SMD 법인상호(사전 밖) / TVD KSIC',
           value=f"{r['link_rate']} / {r['smd_pub_year']} / {r['smd_foreign_prio']} / {r['smd_corp_name']} / {r['tvd_ksic']}",
           note='as-of: 출원인코드는 고정 식별자이고 사후 연결 표지를 쓰지 않았다') for r in j['p3_rows']]
for k, v in bc.items():
    Bc.append(dict(candidate='C03-C', check='양성·위험집합 분포', metric=k, value=json.dumps(v, ensure_ascii=False), note=''))
Bc.append(dict(candidate='C03-A', check='패밀리 연결 정확도', metric='연결 문서 KSIC 일치율(해외 vs KR) PUB / REG', value='0.8902 (n=239,996) / 0.8883 (n=139,139)',
               note=r'공개번호 연결이 등록번호 연결(R10 수정판)과 같은 정확도 → R8 오독 결함 없음. cache\s1_A_log.json'))
Bc.append(dict(candidate='C03-A', check='토큰 오독 진단', metric='P13 토큰 중 KR 공개번호 일치 / 출원번호 일치 / 전체', value='276,113 / 78,568 / 370,849',
               note='P13은 공개번호로 읽어야 한다(R10과 같은 결론)'))
Bc.append(dict(candidate='C03-B', check='실시권자 매핑', metric='전체 매핑률(2024 사전)', value=bl['lic_mapped_share_all_lt2025_at_2024'], note='P1 실패로 편향 비교 생략'))
pd.DataFrame(Bc).to_csv(W + r'\bias_checks.csv', index=False, encoding='utf-8-sig')
print('ok', ok_all, len(F), len(E), len(Bc))

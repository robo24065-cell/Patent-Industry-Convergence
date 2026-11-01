# -*- coding: utf-8 -*-
"""ntis_tables.md 생성 — build_ntis_tables.py 산출 CSV를 그대로 옮긴다(숫자 전사 오류 방지). NTIS Open API 활용."""
import pathlib
import pandas as pd
H = pathlib.Path(__file__).resolve().parent
R = lambda n: pd.read_csv(H / n, encoding='utf-8-sig', dtype={'ksic': str})
pct = lambda v, d=1: '' if pd.isna(v) else f'{v * 100:.{d}f}%'
num = lambda v, d=1: '' if pd.isna(v) else f'{v:,.{d}f}'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(out)


L = []
L.append('# 보고서용 NTIS 기술통계 표 — 정부 R&D × 특허-산업(KSIC) 융합데이터\n')
L.append('**데이터 출처: NTIS Open API 활용**(국가과학기술지식정보서비스 국가R&D 과제검색·성과검색, 2026-09-29 전수 수집, 비상업·내부 분석, 제3자 제공 금지).')
L.append('제공 데이터(KSIC 세세분류 50개 × 5개국 특허)의 KR 특허를 NTIS 성과 특허(출원·등록번호 13자리)와 잇고, 과제(과학기술표준분류)를 성과 특허 경유 배분표로 KSIC에 연결했다.')
L.append('예측 성적과 무관한 기술통계다. 모든 수치는 `build_ntis_tables.py` 실행 결과 CSV(같은 폴더)에서 이 파일로 자동 전사했다. 🔒 최종시험 결과변수는 쓰지 않았다(각 표 주석).\n')
L.append('공통 주의: NTIS는 2026-09 스냅샷이다(성과 등록일 없음 → 준-정합). 성과 특허 연결은 2007년 출원·등록분부터라 2006년 이전은 비교하지 않는다. 연관이지 인과가 아니다.\n')

# 표 1
t = R('ntis_t1_ksic_gov_share.csv')
L.append('## 표 1. KSIC별 정부 R&D 기원 특허 비중과 추이 (`ntis_t1_ksic_gov_share.csv`)')
L.append('제공 KR **특허** 출원(실용신안 제외), 출원연도 2007–2022(5개국 완결 연도). 비중 = NTIS 성과 특허로 연결되는 출원 비율(스냅샷 전체 연결). 한 출원이 여러 KSIC 라벨이면 각 KSIC에 모두 센다. 수행주체 구성은 정부 기원 특허 중 해당 주체 과제가 걸린 비율(복수 가능).\n')
tt = t.copy()
show = pd.DataFrame({'KSIC': tt.ksic, '산업': tt.name, '출원 수': tt.n_2007_22.map(lambda v: f'{v:,}'),
                     '정부 기원 비중 2007–22': tt.gov_share_2007_22.map(pct),
                     '2007–10': tt['gov_2007–10'].map(pct), '2011–14': tt['gov_2011–14'].map(pct), '2015–18': tt['gov_2015–18'].map(pct), '2019–22': tt['gov_2019–22'].map(pct),
                     '변화(%p, 2019–22 − 2007–10)': tt.chg_pp.map(lambda v: f'{v:+.1f}'),
                     '대학': tt.perf_univ.map(pct), '출연연': tt.perf_gri.map(pct), '기업': tt.perf_firm.map(pct)})
L.append(md(show)); L.append('')
tot = t[t.ksic == '전체'].iloc[0]; k = t[t.ksic != '전체']
L.append(f"- 50개 KSIC 전체 비중은 {pct(tot['gov_2007–10'])}(2007–10) → {pct(tot['gov_2019–22'])}(2019–22)로 {tot.chg_pp:+.1f}%p 늘었다.")
top = k.nlargest(3, 'gov_share_2007_22'); up = k.nlargest(3, 'chg_pp'); dn = k.nsmallest(3, 'chg_pp')
L.append('- 비중 상위: ' + ', '.join(f"{r.name}({r.ksic}) {pct(r.gov_share_2007_22)}" for r in top.itertuples()) + '.')
L.append('- 가장 크게 늘어난 산업: ' + ', '.join(f"{r.name} {r.chg_pp:+.1f}%p" for r in up.itertuples()) +
         '. 줄어든 산업: ' + ', '.join(f"{r.name} {r.chg_pp:+.1f}%p" for r in dn.itertuples()) + '.\n')
y = R('ntis_t1b_year_gov_share.csv')
L.append('### 표 1b. 출원연도별 정부 기원 비중(50개 KSIC 합계) (`ntis_t1b_year_gov_share.csv`)\n')
L.append(md(pd.DataFrame({'출원연도': y.appl_year, '출원 수(KSIC 라벨 중복 포함)': y.n.map(lambda v: f'{v:,}'), '정부 기원': y.gov.map(pct),
                          '대학 수행': y.univ.map(pct), '출연연 수행': y.gri.map(pct), '기업 수행': y.firm.map(pct)})))
L.append('- 최근 연도일수록 성과 보고가 덜 쌓였을 수 있다(스냅샷). 2022년 값은 보고 지연으로 더 오를 수 있다.\n')

# 표 2
t2 = R('ntis_t2_lapse_by_performer.csv')
L.append('## 표 2. 수행주체별 3년 내 등록료불납 포기율 (`ntis_t2_lapse_by_performer.csv`)')
L.append('T4 개발 원점 프레임(기준일 L 2012-01 … 2018-07, 결과 창 끝 ≤ 2021-07). 등록 2007년 이후 특허만. 정부 기원은 **성과 레코드 Year ≤ L연도−1**(기준일에 알 수 있던 연결)만.\n')
s2 = pd.DataFrame({'원점': t2.origin, '기준일': t2.L, '특허×기준일': t2.n.map(lambda v: f'{v:,}'), '정부 기원 비중': t2.share_gov.map(pct),
                   '전체': t2.rate_all.map(pct), '정부 기원 아님': t2.rate_nongov.map(pct), '정부 기원': t2.rate_gov.map(pct),
                   '대학': t2.rate_univ.map(pct), '출연연·국공립': t2.rate_gri.map(pct), '대기업·중견': t2.rate_big.map(pct), '중소기업': t2.rate_sme.map(pct)})
L.append(md(s2)); L.append('')
fe = pd.read_csv(H.parent / '_shared' / 'ext' / 'ntis_T_work' / 'ntis_T4_fe.csv', encoding='utf-8-sig')
fe = fe[(fe['sample'] == 'rg2007') & (fe['flags'] == 'g') & (fe['model'] == 'grp')]
nm = {'g_univ': '대학', 'g_gri': '출연연·국공립', 'g_big': '대기업·중견', 'g_sme': '중소기업', 'g_oth': '기타'}
rows = []
for (f_, x), g in fe.groupby(['fe', 'x']):
    g = g.set_index('origin')
    rows.append({'고정효과': f_, '수행주체': nm[x], **{o: f"{g.loc[o, 'coef'] * 100:+.2f} (t {g.loc[o, 't']:.2f})" for o in ['O12', 'O14', 'O16', 'O18']},
                 '평균(%p)': f"{g.coef.mean() * 100:+.2f}", '음(−) 원점': f"{int((g.coef < 0).sum())}/4"})
L.append('### 표 2b. 같은 조건 안의 포기확률 차이(선형확률모형, %p, 출원인 군집 표준오차) — `_shared/ext/ntis_T_work/ntis_T4_fe.csv`에서 옮김\n')
L.append(md(pd.DataFrame(rows)))
L.append('- 원시 비율로는 대학 수행 특허의 포기율이 기준일이 늦을수록 높아진다(2012년 16.0% → 2016년 33.3%). 기업 수행(대기업·중견 17.0–22.3%, 중소 18.3–20.1%)은 정부 기원이 아닌 특허(23.1–25.6%)보다 낮다.')
L.append('- 연차·KSIC·출원인유형을 고정하면 출연연·대기업·중소 수행 특허는 4개 원점 모두 포기확률이 낮다(음). 같은 출원인 안에서 비교하면 대기업·중견 수행분이 가장 크게 남고(3/4 음), 출연연 효과는 사라진다(1/4 음). 대학 효과도 같은 대학 안에서는 0 근처라 "어느 대학이냐"(출원인 간 차이)에서 온다.\n')

# 표 3
t3 = R('ntis_t3_cohort_lapse_transfer.csv')
L.append('## 표 3. 등록 코호트(2007–2016) 5년 내 포기·권리이전 비율 (`ntis_t3_cohort_lapse_transfer.csv`)')
L.append('제공 범위 KR 등록특허(등록원부 `registry_full.sqlite`, 읽기 전용). 포기 = 등록료불납 소멸등록, 이전 = 권리자 이력의 이전 등록(전부·일부·지분). 5년 창 끝 ≤ 2021. 수행주체는 스냅샷 전체 연결(기술통계용)이며 복수 가능.\n')
L.append(md(pd.DataFrame({'구분': t3.group, '특허 수': t3.n.map(lambda v: f'{v:,}'), '5년 내 포기': t3.lapse5.map(pct), '5년 내 이전': t3.transfer5.map(pct),
                          '포기(등록 2007–11)': t3.lapse5_2007_11.map(pct), '포기(2012–16)': t3.lapse5_2012_16.map(pct),
                          '이전(2007–11)': t3.transfer5_2007_11.map(pct), '이전(2012–16)': t3.transfer5_2012_16.map(pct)})))
L.append('- 정부 R&D 기원 특허는 5년 내 포기가 적고(표의 전체 대비), 이전 비율은 비슷하거나 약간 낮다. 대학 수행분은 2012–16 등록 코호트에서 포기가 늘었다(표 2의 원시 추세와 같은 방향).\n')

# 표 4
t4 = R('ntis_t4_entry_by_gov.csv')
L.append('## 표 4. 정부 R&D 이력별 기업×산업 첫 진입률 (`ntis_t4_entry_by_gov.csv`)')
L.append('T1 위험쌍(원점 c에 목표 KSIC 공개·등록 선행출원이 없는 법인×KSIC). 진입 = (c, c+4] 출원이 그 KSIC에 공개·등록. 개발 원점 2008–2012와 보조 원점 2015만(최종시험 c=2019 제외).')
L.append('정부 기원 특허 = 원점 연말까지 보이는 자사 특허(범위안 출원은 공개·등록 ≤ c, 범위밖은 등록 ≤ c) 중 성과 레코드 Year ≤ c. 과제 수행(이름 연결) = NTIS 수행기관명 ↔ 출원인명 정규화 일치, 과제연도 ≤ c−1.\n')
s4 = pd.DataFrame({'원점 c': t4.c, '진입창': t4.window, '위험쌍': t4.pairs.map(lambda v: f'{v:,}'), '전체 진입률': t4.entry_all.map(lambda v: pct(v, 2)),
                   '정부 기원 특허 보유(쌍 비중)': t4.share_g_any.map(pct), '보유 진입률': t4.entry_g_any.map(lambda v: pct(v, 2)), '미보유': t4.entry_not_g_any.map(lambda v: pct(v, 2)),
                   '배수': (t4.entry_g_any / t4.entry_not_g_any).map(lambda v: f'{v:.1f}'),
                   '기업 직접 수행 보유': t4.entry_g_self.map(lambda v: pct(v, 2)), '대학·출연연 공동 보유': t4.entry_g_coop.map(lambda v: pct(v, 2)),
                   '과제분류→목표 KSIC 리프트 상위 25%': t4.entry_gtgt_top25.map(lambda v: pct(v, 2)), '하위 25%': t4.entry_gtgt_bot25.map(lambda v: pct(v, 2)),
                   '과제 수행(이름 연결)': t4.entry_q_any.map(lambda v: pct(v, 2)), '미수행': t4.entry_not_q_any.map(lambda v: pct(v, 2))})
L.append(md(s4))
L.append('- 정부 기원 특허를 가진 기업의 진입률은 1.9–3.0배다. 보유 기업 안에서도 **과제분류가 목표 산업 쪽을 가리키는(리프트 상위 25%)** 쌍은 하위 25%보다 진입률이 3.4–4.8배 높다. 이 신호(g_tgt_cls)가 T1 2단 보정층 R4E01로 채택됐다(개발 skill 0.3469 → 0.3516, 5/5).')
L.append('- 과제 수행 여부만으로는(이름 연결) 진입률 차이가 작다. 특허로 이어진 과제, 그리고 과제의 분야 방향이 신호다.')
L.append('- 사업자번호로 연결한 과제 수행 변수는 사후 목록(CORP_APPLICANT 2026-07) 생존 편향이 확인돼 이 표에서 뺐다(T1 R4E04 비고).\n')

# 표 5·6
t5 = R('ntis_t5_funds_by_ksic.csv')
L.append('## 표 5. 정부 R&D 연구비의 산업(KSIC)별 배분 추이 (`ntis_t5_funds_by_ksic.csv`)')
L.append('과제 정부연구비를 과학기술표준분류 → KSIC 배분표(분류별 KR 특허 성과 중 KSIC 비중, **성과 Year ≤ t−2**, 중분류→대분류 수축)로 나눈 값(`ntis_project_ksic_year.parquet`, method a_link_pit, 2009–2023). 연평균 십억원. 50개 KSIC 밖 분야로 가는 몫이 있어 50개 합계는 연도 전체의 22–25%다.\n')
s5 = pd.DataFrame({'KSIC': t5.ksic, '산업': t5.name, '2009–12': t5['funds_bil_2009–12'].map(num), '2013–16': t5['funds_bil_2013–16'].map(num),
                   '2017–20': t5['funds_bil_2017–20'].map(num), '2021–23': t5['funds_bil_2021–23'].map(num),
                   '50개 중 비중 2009–12': t5['share50_2009–12'].map(pct), '50개 중 비중 2021–23': t5['share50_2021–23'].map(pct),
                   '연평균 증가율': t5.cagr_2009_12_to_2021_23.map(pct),
                   '수행 구성 2017–23 기업': t5.firm_2017_23.map(pct), '대학': t5.univ_2017_23.map(pct), '출연연': t5.gri_2017_23.map(pct)})
L.append(md(s5))
L.append('- 연평균 증가율 = (2021–23 연평균 / 2009–12 연평균)^(1/11.5) − 1.\n')
t6 = R('ntis_t6_funds_year.csv')
L.append('## 표 6. 연도별 50개 KSIC 배분 정부연구비와 수행주체 구성 (`ntis_t6_funds_year.csv`)\n')
L.append(md(pd.DataFrame({'연도': t6.year, '50개 KSIC 배분(십억원)': t6.gov_50_bil.map(num), '연도 전체 정부연구비(십억원, 수집분)': t6.year_total_bil.map(num),
                          '50개 비중': t6.share50_of_nat.map(pct), '기업 수행': t6.firm_share.map(pct), '대학': t6.univ_share.map(pct), '출연연': t6.gri_share.map(pct)})))
L.append('- 수집 과제 합계는 공식 국가R&D 총액보다 작다(보안과제 등 OpenAPI 미제공분). 2024년은 등록 지연으로 제외했다.\n')
L.append('## 파일')
L.append('- 생성 스크립트: `build_ntis_tables.py`(CSV), `build_ntis_md.py`(이 문서). 입력은 모두 읽기 전용.')
L.append('- CSV: `ntis_t1_ksic_gov_share.csv`, `ntis_t1b_year_gov_share.csv`, `ntis_t2_lapse_by_performer.csv`, `ntis_t3_cohort_lapse_transfer.csv`, `ntis_t4_entry_by_gov.csv`, `ntis_t5_funds_by_ksic.csv`, `ntis_t6_funds_year.csv`')
L.append('- 출처 표기(보고서·그림 공통): "NTIS Open API 활용(국가과학기술지식정보서비스)".')
(H / 'ntis_tables.md').write_text('\n'.join(L), encoding='utf-8')
print('ok', len(L))

# (d) 성과 보고 지연 곡선과 트랙별 '아직 보고 안 된 양성' 추정
# 근거 3개: (1) NTIS 스냅샷(2026-09-29) 성과연도별 KR 특허 고유 건수, (2) 공식 조사·분석 2022 성과(국내특허 출원 36,822·등록 15,983, KIST 2022 융합연구 성과분석 표 인용)
#           (3) 제공 KR 출원 중 NTIS 연결률(출원연도별, ntis_link_rates.csv). 과제 등록 완결도는 공식 과제 수와 대조(_tab_a_totals.csv).
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
EXT = os.path.join(MC, r"_shared\ext")
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T1"
SNAP = 2026 + (272 / 365)   # 2026-09-29
r = pd.read_parquet(os.path.join(EXT, 'ntis_rpatent_raw.parquet'), columns=['Year', 'RegistType', 'IprType', 'is_kr', 'ProjectYear', 'RegistNumber', 'ProjectID'])
r = r[r.is_kr.fillna(False).astype(bool) & (r.IprType == '특허')].copy()
r['Year'] = pd.to_numeric(r.Year, errors='coerce'); r['PY'] = pd.to_numeric(r.ProjectYear, errors='coerce')
rows = []
for y in range(2007, 2027):
    a = r[(r.Year == y) & (r.RegistType == '출원')]; g = r[(r.Year == y) & (r.RegistType == '등록')]
    rows.append(dict(year=y, kr_appl_records=len(a), kr_appl_unique=a.RegistNumber.nunique(), kr_rgst_records=len(g), kr_rgst_unique=g.RegistNumber.nunique(),
                     yrs_since_mid_year=round(SNAP - (y + 0.5), 2)))
C = pd.DataFrame(rows)
lr = pd.read_csv(os.path.join(EXT, 'ntis_link_rates.csv'))
lr = lr[lr.level == 'year'][['appl_year', 'n_patents', 'n_linked', 'link_rate']].rename(columns={'appl_year': 'year'})
lr['year'] = lr.year.astype(int)
C = C.merge(lr, on='year', how='left')
ref_lr = lr[lr.year.between(2019, 2021)].n_linked.sum() / lr[lr.year.between(2019, 2021)].n_patents.sum()
ref_cnt = C[C.year.between(2019, 2021)].kr_appl_unique.mean()
C['link_rate_rel_2019_21'] = C.link_rate / ref_lr
C['appl_count_rel_2019_21'] = C.kr_appl_unique / ref_cnt
# 공식 대조 (2022 성과연도, 조사·분석 집계 시점 ≈ 2023년 상반기 → 출원 후 평균 약 0.75–1.0년)
OFF22 = dict(appl=36822, rgst=15983)
s22 = C.set_index('year').loc[2022]
off = dict(official_2022_appl=OFF22['appl'], snapshot_2022_appl_unique=int(s22.kr_appl_unique), snapshot_2022_appl_records=int(s22.kr_appl_records),
           ratio_appl_unique=OFF22['appl'] / s22.kr_appl_unique, ratio_appl_records=OFF22['appl'] / s22.kr_appl_records,
           official_2022_rgst=OFF22['rgst'], snapshot_2022_rgst_unique=int(s22.kr_rgst_unique), ratio_rgst_unique=OFF22['rgst'] / s22.kr_rgst_unique)
print(json.dumps(off, ensure_ascii=False, default=float))
# 과제연도 코호트 × 출원 오프셋: 비보안 과제당 KR 출원 성과(고유 특허×과제)
P = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['pjt_no', 'year', 'secret_flag'])
npj = P[~P.secret_flag].groupby('year').size()
a = r[r.RegistType == '출원'].drop_duplicates(['ProjectID', 'RegistNumber'])
a['k'] = a.Year - a.PY
coh = a[a.PY.between(2012, 2024) & a.k.between(-2, 3)].groupby(['PY', 'k']).size().unstack(fill_value=0)
coh = coh.div(npj.reindex(coh.index), axis=0) * 1000   # 과제 1,000건당
print('cohort per 1000 projects'); print(coh.round(1).to_string())
# 곡선 점: (출원 후 경과연수, 완결도 추정)
pts = [dict(source='official_survey_2022', s=0.85, completeness=off['ratio_appl_unique'], note='조사·분석 집계(2023 상반기 추정) / 2026-09 스냅샷 고유건수')]
for y in (2021, 2022, 2023, 2024, 2025):
    row = C.set_index('year').loc[y]
    pts.append(dict(source=f'link_rate_{y}', s=float(row.yrs_since_mid_year), completeness=float(row.link_rate_rel_2019_21), note=f'제공 공개 KR 출원 {int(row.n_patents)}건 중 연결률 / 2019–21'))
    pts.append(dict(source=f'ntis_count_{y}', s=float(row.yrs_since_mid_year), completeness=float(row.appl_count_rel_2019_21), note='NTIS KR 출원 성과 고유건수 / 2019–21 평균(예산·제도 변화 혼재)'))
LC = pd.DataFrame(pts)
out = pd.concat([C.assign(table='by_year'), LC.assign(table='curve_points'), coh.reset_index().melt(id_vars='PY', var_name='k', value_name='per_1000_projects').assign(table='cohort_offset')], ignore_index=True)
out.to_csv(os.path.join(OUT, 'lag_curve.csv'), index=False, encoding='utf-8-sig')
print(C.round(3).to_string()); print(LC.round(3).to_string())
json.dump(dict(official_cmp=off, ref_link_rate_2019_21=ref_lr, ref_count_2019_21=ref_cnt), open(os.path.join(OUT, '_tab_d_lag.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)

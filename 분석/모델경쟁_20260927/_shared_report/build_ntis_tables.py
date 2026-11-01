# -*- coding: utf-8 -*-
"""보고서용 NTIS 기술통계 표 (예측 성적과 무관). 출처: NTIS Open API(국가과학기술지식정보서비스 과제·성과 검색) 활용 — 비상업·내부 분석, 제3자 제공 금지.
입력은 모두 읽기만 한다: _shared/ext/ntis_*, 수집파이프라인/registry_full.sqlite(mode=ro), T4 R4 개발 프레임(gv_ 피처), T1 R4 캐시·R3 동결 개발 예측의 레이블.
출력: _shared_report/ntis_t{1..6}_*.csv (표 본문은 ntis_tables.md 에 옮겨 적는다)
🔒 최종시험 결과변수는 쓰지 않는다: T1 진입은 개발 원점 c ≤ 2012 와 보조 원점 2015(진입 2016–19)만, T4 포기는 개발 원점 O12–O18(결과 ≤2021-07)만,
   등록 코호트 표는 등록 2007–2016(5년 창 끝 ≤2021). 출원연도 표는 5개국 완결 연도인 ≤2022.
"""
import os, json, sqlite3, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent; BASE = ROOT.parent.parent
EXTD = ROOT / '_shared' / 'ext'
OUT = lambda n: HERE / n
GRPN = {'univ': '대학', 'gri': '출연연·국공립연', 'big': '대기업·중견', 'sme': '중소기업', 'oth': '기타·부처'}
kn = {d['code']: d['codenmKr'] for d in json.load(open(BASE / '공모전데이터' / '외부데이터' / 'KSIC해설' / 'ksic50_definitions.json', encoding='utf-8'))}

# ---------------- 표 1. KSIC별 정부R&D 기원 특허 비중과 추이 (제공 KR 특허 출원, 출원연도 2007–2022) ----------------
f = pd.read_parquet(EXTD / 'ntis_patent_flag.parquet', columns=['appl_no', 'appl_year', 'ksic', 'gov_rnd', 'has_univ', 'has_gri', 'has_firm'])
f = f[f.appl_no.astype(str).str[:2] == '10'].copy()                        # 특허만(실용신안 제외)
f['appl_year'] = pd.to_numeric(f.appl_year, errors='coerce'); f = f[(f.appl_year >= 2007) & (f.appl_year <= 2022)]
f['ksic'] = f.ksic.astype(str)
for c in ('gov_rnd', 'has_univ', 'has_gri', 'has_firm'):
    f[c] = f[c].fillna(False).astype(bool)
P = {'2007–10': (2007, 2010), '2011–14': (2011, 2014), '2015–18': (2015, 2018), '2019–22': (2019, 2022)}
rows = []
for k, g in list(f.groupby('ksic')) + [('전체', f)]:
    r = dict(ksic=k, name=kn.get(k, '50개 KSIC 합계(중복 라벨 포함)'), n_2007_22=len(g), gov_share_2007_22=g.gov_rnd.mean())
    for p, (a, b) in P.items():
        s = g[(g.appl_year >= a) & (g.appl_year <= b)]; r[f'gov_{p}'] = s.gov_rnd.mean() if len(s) else np.nan
    r['chg_pp'] = (r['gov_2019–22'] - r['gov_2007–10']) * 100
    gg = g[g.gov_rnd]
    r['perf_univ'] = gg.has_univ.mean(); r['perf_gri'] = gg.has_gri.mean(); r['perf_firm'] = gg.has_firm.mean()
    rows.append(r)
T1 = pd.DataFrame(rows)
T1 = pd.concat([T1[T1.ksic != '전체'].sort_values('gov_share_2007_22', ascending=False), T1[T1.ksic == '전체']])
T1.to_csv(OUT('ntis_t1_ksic_gov_share.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
Y = f.groupby('appl_year').agg(n=('gov_rnd', 'size'), gov=('gov_rnd', 'mean'), univ=('has_univ', 'mean'), gri=('has_gri', 'mean'), firm=('has_firm', 'mean')).reset_index()
Y.to_csv(OUT('ntis_t1b_year_gov_share.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('표1', len(T1), flush=True)

# ---------------- 표 2. 수행주체별 3년 내 포기율 (T4 개발 원점 프레임, 기준일 L 2012-01~2018-07, 등록 ≥2007, 성과 레코드 Year ≤ L연도−1) ----------------
T4R = ROOT / 'T4_특허포기예측' / 'R4' / 'runs'
rows = []
for o in ['O12', 'O14', 'O16', 'O18']:
    ev = pd.read_parquet(T4R / o / 'ev.parquet', columns=['pid', 'L', 'y', 'rgst_year'])
    for g in GRPN:
        ev[g] = np.load(T4R / o / 'x' / f'ev__gv_{g}.npy')
    e = ev[ev.y.notna() & (ev.rgst_year >= 2007)]
    anyg = e[list(GRPN)].max(1) == 1
    r = dict(origin=o, L=','.join(sorted(e.L.astype(str).str[:7].unique())), n=len(e), rate_all=e.y.mean(), rate_nongov=e.y[~anyg].mean(), rate_gov=e.y[anyg].mean(),
             share_gov=anyg.mean())
    for g in GRPN:
        m = e[g] == 1; r[f'rate_{g}'] = e.y[m].mean(); r[f'n_{g}'] = int(m.sum())
    rows.append(r)
T2 = pd.DataFrame(rows); T2.to_csv(OUT('ntis_t2_lapse_by_performer.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
fe = pd.read_csv(EXTD / 'ntis_T_work' / 'ntis_T4_fe.csv', encoding='utf-8-sig')
print('표2', flush=True)

# ---------------- 표 3. 등록 코호트(2007–2016) 5년 내 포기·권리이전 비율 — 등록원부(registry_full, 읽기 전용) ----------------
con = sqlite3.connect(f"file:{(BASE / '수집파이프라인' / 'registry_full.sqlite').as_posix()}?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, rgstDate, ksic from patent where substr(rgst_no,1,2)='10'", con)
lap = pd.read_sql("select rgst_no, min(rgstCsDate) d from right where rgstCsName='소멸등록' and rgstCsReason like '%불납%' group by rgst_no", con)
trf = pd.read_sql("select rgst_no, min(ownerRgstCsDate) d from owner where ownerRgstCsReason like '%이전%' and ownerRgstCsDate is not null group by rgst_no", con)
con.close()
pt['rd'] = pd.to_datetime(pt.rgstDate.astype(str).str[:8], format='%Y%m%d', errors='coerce'); pt = pt[pt.rd.notna()]
pt = pt[(pt.rd.dt.year >= 2007) & (pt.rd.dt.year <= 2016)].drop_duplicates('rgst_no')
for nm, t in (('lap', lap), ('trf', trf)):
    t['d'] = pd.to_datetime(t.d.astype(str).str[:8], format='%Y%m%d', errors='coerce')
    pt = pt.merge(t.rename(columns={'d': nm}), on='rgst_no', how='left')
    pt[f'{nm}5'] = (pt[nm].notna() & (pt[nm] <= pt.rd + pd.DateOffset(years=5))).astype(float)
fl = pd.read_parquet(EXTD / 'ntis_patent_flag.parquet', columns=['rgst_no', 'gov_rnd', 'has_univ', 'has_gri', 'has_firm']).dropna(subset=['rgst_no'])
fl['rgst_no'] = fl.rgst_no.astype(str); fl = fl.groupby('rgst_no')[['gov_rnd', 'has_univ', 'has_gri', 'has_firm']].max().reset_index()
pt['rgst_no'] = pt.rgst_no.astype(str); pt = pt.merge(fl, on='rgst_no', how='left')
for c in ('gov_rnd', 'has_univ', 'has_gri', 'has_firm'):
    pt[c] = pt[c].fillna(False).astype(bool)
grp = {'전체': np.ones(len(pt), bool), '정부R&D 기원 아님': ~pt.gov_rnd.values, '정부R&D 기원(전체)': pt.gov_rnd.values,
       '대학 수행': pt.has_univ.values, '출연연·국공립연 수행': pt.has_gri.values, '기업 수행': pt.has_firm.values}
rows = []
for nm, m in grp.items():
    s = pt[m]
    rows.append(dict(group=nm, n=len(s), lapse5=s.lap5.mean(), transfer5=s.trf5.mean(),
                     lapse5_2007_11=s[s.rd.dt.year <= 2011].lap5.mean(), lapse5_2012_16=s[s.rd.dt.year >= 2012].lap5.mean(),
                     transfer5_2007_11=s[s.rd.dt.year <= 2011].trf5.mean(), transfer5_2012_16=s[s.rd.dt.year >= 2012].trf5.mean()))
T3 = pd.DataFrame(rows); T3.to_csv(OUT('ntis_t3_cohort_lapse_transfer.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('표3', len(pt), flush=True)

# ---------------- 표 4. 수행주체·정부과제 이력별 기업×산업 첫 진입률 (T1 위험쌍, 개발 원점 2008–2012 + 보조 2015) ----------------
T1R = ROOT / 'T1_산업대기실'
rows = []
for c in [2008, 2009, 2010, 2011, 2012, 2015]:
    o = pd.read_parquet(T1R / 'R3' / 'runs' / 'FROZEN_dev' / f'o{c}.parquet', columns=['firm', 'k', 'y_u', 'prior_any'])
    o = o[~o.prior_any.values].reset_index(drop=True)
    a = pd.read_parquet(T1R / 'R4' / 'cache' / f'ntis_o{c}.parquet'); b = pd.read_parquet(T1R / 'R4' / 'cache' / f'ntis_name_o{c}.parquet')
    assert (a.firm.values == o.firm.astype(str).values).all()
    y = o.y_u.values
    r = dict(c=c, window=f'{c + 1}–{c + 4}', pairs=len(o), entry_all=y.mean())
    for nm, m in (('g_any', a.g_any.values == 1), ('g_self', a.g_self.values == 1), ('g_coop', a.g_coop.values == 1), ('q_any', b.q_any.values == 1)):
        r[f'share_{nm}'] = m.mean(); r[f'entry_{nm}'] = y[m].mean(); r[f'entry_not_{nm}'] = y[~m].mean()
    gg = a.g_any.values == 1
    if np.isfinite(a.g_tgt_cls.values[gg]).any():
        q = pd.Series(a.g_tgt_cls.values[gg]).rank(pct=True).values
        r['entry_gtgt_top25'] = y[gg][q > 0.75].mean(); r['entry_gtgt_bot25'] = y[gg][q <= 0.25].mean()
    rows.append(r)
T4 = pd.DataFrame(rows); T4.to_csv(OUT('ntis_t4_entry_by_gov.csv'), index=False, encoding='utf-8-sig', float_format='%.5f')
print('표4', flush=True)

# ---------------- 표 5·6. 정부 R&D 연구비의 산업(KSIC)별 배분 추이 (과제→성과 특허 배분표, 성과 Year ≤ t−2, a_link_pit) ----------------
kq = pd.read_parquet(EXTD / 'ntis_project_ksic_year.parquet'); kq['ksic'] = kq.ksic.astype(str)
a = kq[(kq.method == 'a_link_pit') & (kq.year <= 2023)].copy()
PP = {'2009–12': (2009, 2012), '2013–16': (2013, 2016), '2017–20': (2017, 2020), '2021–23': (2021, 2023)}
rows = []
tot_all = {p: a[(a.year >= lo) & (a.year <= hi)].gov_funds.sum() for p, (lo, hi) in PP.items()}
yt = a.drop_duplicates('year').set_index('year').year_total_gov
for k, g in a.groupby('ksic'):
    r = dict(ksic=k, name=kn.get(k, ''))
    for p, (lo, hi) in PP.items():
        s = g[(g.year >= lo) & (g.year <= hi)]
        r[f'funds_bil_{p}'] = s.gov_funds.sum() / 1e9 / (hi - lo + 1)                 # 연평균 정부연구비(십억원)
        r[f'share50_{p}'] = s.gov_funds.sum() / tot_all[p]                            # 50개 KSIC 배분액 중 비중
        r[f'share_nat_{p}'] = s.gov_funds.sum() / yt.loc[lo:hi].sum()                 # 연도 전체 정부연구비 중 비중
    s = g[g.year >= 2017]; tf = s.gov_funds.sum()
    r['firm_2017_23'] = s.gov_funds_firm.sum() / tf; r['univ_2017_23'] = s.gov_funds_univ.sum() / tf; r['gri_2017_23'] = s.gov_funds_gri.sum() / tf
    r['cagr_2009_12_to_2021_23'] = (r['funds_bil_2021–23'] / r['funds_bil_2009–12']) ** (1 / 11.5) - 1 if r['funds_bil_2009–12'] > 0 else np.nan
    rows.append(r)
T5 = pd.DataFrame(rows).sort_values('funds_bil_2021–23', ascending=False)
T5.to_csv(OUT('ntis_t5_funds_by_ksic.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
Z = a.groupby('year').agg(gov_50=('gov_funds', 'sum'), firm=('gov_funds_firm', 'sum'), univ=('gov_funds_univ', 'sum'), gri=('gov_funds_gri', 'sum'),
                          year_total_gov=('year_total_gov', 'first')).reset_index()
Z['share50_of_nat'] = Z.gov_50 / Z.year_total_gov
for c in ('firm', 'univ', 'gri'):
    Z[c + '_share'] = Z[c] / Z.gov_50
Z['gov_50_bil'] = Z.gov_50 / 1e9; Z['year_total_bil'] = Z.year_total_gov / 1e9
Z[['year', 'gov_50_bil', 'year_total_bil', 'share50_of_nat', 'firm_share', 'univ_share', 'gri_share']].to_csv(OUT('ntis_t6_funds_year.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
print('표5·6', flush=True)
json.dump(dict(source='NTIS Open API 활용(국가과학기술지식정보서비스 과제검색·성과검색)', note='비상업·내부 분석, 제3자 제공 금지',
               fe_t4_file=str(EXTD / 'ntis_T_work' / 'ntis_T4_fe.csv')), open(OUT('ntis_tables_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

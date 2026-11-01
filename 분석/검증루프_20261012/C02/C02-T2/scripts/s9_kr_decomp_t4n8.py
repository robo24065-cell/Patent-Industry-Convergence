# C02-T2 s9: KR 2023→2024 등록 감소 분해(제공=등록원부) + 공식 분해 대조 + 법인 포트폴리오로 확인한 '등록 사실 누락' 분해
#            + T4·N8·4.3 단축창 영향 → t4_n8_impact.json  (읽기 전용, 모형·레이블 재계산 없음; s4 캐시는 등록원부와 재대조 후 사용)
import os, sys, json, sqlite3
import duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2"); C = os.path.join(W, 'cache')
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
REG = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
res = {}
# ---- 1) 등록원부 슬림 캐시 재대조
r = pd.read_parquet(os.path.join(C, 's1_registry_slim.parquet'))
rc = sqlite3.connect(f'file:{REG}?mode=ro', uri=True)
chk = pd.read_sql("select substr(rgstDate,1,4) y, count(*) n from patent where rgstDate>='2019' group by 1", rc).set_index('y').n
rc.close()
res['slim_vs_registry_year_equal'] = bool((r[r.y >= '2019'].groupby('y').size().reindex(chk.index) == chk).all())
r = r[r.rgst_no.str[:2] == '10']
r['old'] = np.where(r.ay.astype(int) <= 2019, '구출원(≤2019)', '신출원(2020+)')
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'"); con.sql("install sqlite; load sqlite")
pr = con.sql(f"""select rgst_no, max(case when prio_cc is not null and regexp_replace(replace(prio_cc,'KR',''),'\\|','','g')<>'' then 1 else 0 end) prio_for
                 from read_parquet('{META}') where cc='KR' and rgst_no is not null group by 1""").df()
r = r.merge(pr, on='rgst_no', how='left'); r['prio'] = np.where(r.prio_for == 1, '해외우선권', '우선권없음/국내')
def decomp(df, col):
    a = df[df.y == '2023'].groupby(col).size(); b = df[df.y == '2024'].groupby(col).size()
    t = pd.DataFrame({'n2023': a, 'n2024': b}).fillna(0); t['delta'] = t.n2024 - t.n2023
    t['contrib'] = t.delta / t.delta.sum(); t['rate'] = t.n2024 / t.n2023 - 1; t['share2023'] = t.n2023 / t.n2023.sum()
    return t
dec = {}
for col in ['old', 'natl', 'atype', 'prio', 'ksic']:
    t = decomp(r, col); dec[col] = t
    res.setdefault('decomp', {})[col] = {'max_contrib': float(t.contrib.max()), 'max_cat': str(t.contrib.idxmax()),
                                         'concentrated_ge60': bool(t.contrib.max() >= 0.60),
                                         'max_excess_contrib_minus_share': float((t.contrib - t.share2023).abs().max()),
                                         'table': t.round(4).reset_index().to_dict('records') if col != 'ksic' else None}
kd = dec['ksic']; res['decomp']['ksic']['rate_min'] = float(kd.rate.min()); res['decomp']['ksic']['rate_max'] = float(kd.rate.max())
res['decomp']['ksic']['rate_median'] = float(kd.rate.median()); res['decomp']['ksic']['n_ksic_rate_lt_minus10'] = int((kd.rate < -0.10).sum())
# 공식(KOSIS) 내외국인·개인법인 증감
k = pd.read_csv(os.path.join(C, 's5_kosis_long.csv'), dtype=str); k['DT'] = pd.to_numeric(k.DT, errors='coerce')
a46 = k[(k.src == 'data_TX_13801_A046') & (k.C2_NM == '특허')]
off = {nm: float(a46[(a46.C1_NM == nm) & (a46.PRD_DE == '2024')].DT.iloc[0] / a46[(a46.C1_NM == nm) & (a46.PRD_DE == '2023')].DT.iloc[0] - 1) for nm in ['내국인', '외국인']}
a40 = k[(k.src == 'data_TX_13801_A040_1') & (k.C1_NM == '특허') & (k.C3_NM == '계')]
for nm in ['개인', '법인']:
    off[nm] = float(a40[(a40.C2_NM == nm) & (a40.PRD_DE == '2024')].DT.iloc[0] / a40[(a40.C2_NM == nm) & (a40.PRD_DE == '2023')].DT.iloc[0] - 1)
off['특허 전체'] = float(134734 and (127806 / 134734 - 1))
res['official_change_2023_2024'] = off
# ---- 2) 법인 포트폴리오(2026 스냅샷)로 확인한 누락 등록: 제공에 출원은 있고 rgst 공란, br 에서 등록(≤2025-08-22)
con.sql(f"""create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(appl_year) ay, any_value(ksic) ksic, any_value(pub_date) pub_date,
              max(case when prio_cc is not null and regexp_replace(replace(prio_cc,'KR',''),'\\|','','g')<>'' then 1 else 0 end) prio_for
            from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10' group by appl_no""")
con.sql(f"""create temp table br as select appl_no, min(rgst_date) rd from sqlite_scan('{BR}', 'right_list') where substr(rgst_no,1,2)='10' group by appl_no""")
ms = con.sql("""select left(b.rd,6) ym, k.ay, k.ksic, k.prio_for from kr k join br b using(appl_no)
                where (k.rgst_no is null or k.rgst_no='') and b.rd<='20250822'""").df()
cov = con.sql("""select avg((b.appl_no is not null)::int) from kr k left join br b using(appl_no) where substr(k.rgst_no,1,2)='10' and k.ay between 2018 and 2023""").fetchone()[0]
con.close()
ms['y'] = ms.ym.str[:4]
res['missing_br'] = dict(total=int(len(ms)), by_year=ms.groupby('y').size().to_dict(), br_cover_recent=float(cov),
                         months_with_missing=sorted(ms.ym.unique().tolist()),
                         est_total_by_year={y: float(v / cov) for y, v in ms.groupby('y').size().items()},
                         share_old_appl_2024=float((ms[ms.y == '2024'].ay <= 2019).mean()),
                         share_prio_for_2024=float(ms[ms.y == '2024'].prio_for.mean()))
# 누락분 KSIC 분포 vs 제공 2024 등록 KSIC 분포 (치우침)
m24 = ms[ms.y == '2024'].ksic.value_counts(normalize=True); p24 = r[r.y == '2024'].ksic.value_counts(normalize=True)
res['missing_br']['ksic_tvd_missing_vs_present_2024'] = float(0.5 * (m24.reindex(p24.index.union(m24.index)).fillna(0) - p24.reindex(p24.index.union(m24.index)).fillna(0)).abs().sum())
kk = pd.DataFrame({'present': r[r.y == '2024'].ksic.value_counts(), 'missing_br': ms[ms.y == '2024'].ksic.value_counts()}).fillna(0)
kk['missing_rate_est'] = (kk.missing_br / cov) / (kk.present + kk.missing_br / cov)
res['missing_br']['ksic_missing_rate_2024'] = dict(min=float(kk.missing_rate_est.min()), median=float(kk.missing_rate_est.median()), max=float(kk.missing_rate_est.max()))
# ---- 3) T4·N8 영향 (s4 캐시 + 위 추정)
s4 = json.load(open(os.path.join(C, 's4_out.json'), encoding='utf-8'))
mm = ms[(ms.ym >= '202311') & (ms.ym <= '202412')]
miss_n8 = len(mm) / cov
t4n8 = dict(
    T4_official=dict(population='L=2022-01-01 생존 등록특허(2022 이전 등록)', affected_by_2023_11_plus_gap=False,
                     note='결손 구간(등록 2023-11~2024-08, 2025-02~08)은 모집단 등록일 밖. 결과(등록료 불납 소멸)는 등록원부 pay/right로 관측되어 제공 등록 플래그와 무관'),
    N8_official=dict(population='t0=2022 말 생존 등록특허', event_window='2023-01~2024-12', affected_by_gap=False,
                     note='모집단은 2022년까지 등록분. 결손 등록분(2023-11 이후)은 모집단 밖'),
    T4_short_window=dict(window='납부완료일 2025-01-02~08-31', pop=s4['t4_short_pop'], pop_rg_ge_2023_11=s4['t4_short_pop_rg_ge_2023_11'],
                         share_paidthru_lt_rg_plus3y_rg2021_2023=s4['share_paidthru_lt_rg_plus3y_rg2021_2023'],
                         affected_share=0.0, note='신규 등록은 설정등록 때 3년분 납부 → 2023-11 이후 등록분의 첫 납부완료일은 2026-11 이후라 모집단에 들어올 수 없음'),
    N8_short_window=dict(window='t0=2024 말 생존, 예고등록 2025-01~2026-07', pop_present=s4['n8_pop_t0_2024_approx'],
                         missing_reg_br_2023_11_to_2024_12=int(len(mm)), br_cover=float(cov), missing_est=float(miss_n8),
                         missing_share_of_pop=float(miss_n8 / (s4['n8_pop_t0_2024_approx'] + miss_n8)),
                         young_cohort_rate_official_analog=s4['n8_official_analog_young_rate'],
                         expected_missing_events=float(miss_n8 * s4['n8_official_analog_young_rate']),
                         observed_events=s4['n8_short_pos_observed'],
                         expected_missing_event_share=float(miss_n8 * s4['n8_official_analog_young_rate'] / (s4['n8_short_pos_observed'] + miss_n8 * s4['n8_official_analog_young_rate'])),
                         young_cohort_present_2023_11_to_2024_08=s4['n8_short_pop_rg_2023_11_to_2024_08_present'],
                         young_cohort_present_2022_11_to_2023_08=s4['n8_short_pop_rg_2022_11_to_2023_08_present']),
    official_expected_shortfall_2024=dict(method='r_bar(2019–23)×KIPO 2024 − 제공 2024', value=float(0.25917168265269186 * 127806 - 26766)),
)
t4n8['N8_short_window']['verdict'] = '단서 필요' if t4n8['N8_short_window']['missing_share_of_pop'] > 0.05 else '5% 이내(단서 불요, 기록만)'
t4n8['T4_short_window']['verdict'] = '영향 없음'
res['t4_n8'] = t4n8
json.dump(t4n8 | {'kr_gap_decomposition': res['decomp'], 'official_change_2023_2024': off, 'missing_br': res['missing_br'],
                  'slim_vs_registry_year_equal': res['slim_vs_registry_year_equal']},
          open(os.path.join(W, 't4_n8_impact.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
for col in ['old', 'natl', 'atype', 'prio']:
    print(col); print(dec[col].round(3).to_string())
print('ksic', res['decomp']['ksic'] | {'table': None})
print(off); print({k: v for k, v in res['missing_br'].items() if k != 'months_with_missing'}, res['missing_br']['months_with_missing'])
print(json.dumps(t4n8, ensure_ascii=False, indent=1, default=float))

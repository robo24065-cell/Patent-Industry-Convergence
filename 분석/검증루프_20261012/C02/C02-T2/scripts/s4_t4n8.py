# C02-T2 s4: KR 등록 사실 누락(2023-11~2024-08, 2025-02~08)이 T4·N8 공식 모집단과 4.3 단축창에 주는 영향 (레이블·모형 없음, 건수만)
import os, json, sys, sqlite3
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
REG = os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/")
PB = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache\pat_base.parquet")   # C01 산출(등록원부에서 만든 납부완료일), 아래에서 일부 재검증
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\cache")
res = {}
p = pd.read_parquet(PB, columns=['rgst_no', 'rg', 'max_annl', 'paid_thru', 'lapse_reg_d', 'restore_reg_d', 'expire_reg_d', 'cndrtExptnDate_d'])
# 재검증: 등록원부 patent.rgstDate 와 rg 일치, 등록 시 최소 납부 연수
rc = sqlite3.connect(f'file:{REG}?mode=ro', uri=True)
pt = pd.read_sql("select rgst_no, rgstDate from patent", rc)
m = p.merge(pt, on='rgst_no'); res['rg_match'] = float((m.rg.dt.strftime('%Y%m%d') == m.rgstDate).mean())
rec = p[(p.rg >= '2021-01-01') & (p.rg <= '2023-10-31')]
res['max_annl_dist_rg2021_2023'] = rec.max_annl.value_counts().sort_index().head(8).to_dict()
res['share_paidthru_lt_rg_plus3y_rg2021_2023'] = float((rec.paid_thru < rec.rg + pd.DateOffset(years=3) - pd.Timedelta(days=1)).mean())
# T4 단축창 모집단: 납부완료일 2025-01-02~2025-08-31, (근사) 그때까지 소멸·만료 기록 없음
w = p[(p.paid_thru >= '2025-01-02') & (p.paid_thru <= '2025-08-31')]
res['t4_short_pop'] = int(len(w))
res['t4_short_pop_rg_ge_2023_11'] = int((w.rg >= '2023-11-01').sum())
res['t4_short_pop_rg_2022'] = int(w.rg.dt.year.eq(2022).sum())
res['t4_short_pop_by_rgyear'] = w.rg.dt.year.value_counts().sort_index().tail(6).to_dict()
# T4 공식: L=2022-01-01 살아 있는 특허 → 2022 이전 등록만
res['t4_official_rg_max_le_L'] = True
# N8 모집단 (t0 말까지 등록, 소멸(회복 제외)·만료 없음, 존속만료 > t0 말)
def alive(t0):
    T = pd.Timestamp(f'{t0}-12-31')
    lap = p.lapse_reg_d.notna() & (p.lapse_reg_d <= T) & ~(p.restore_reg_d.notna() & (p.restore_reg_d >= p.lapse_reg_d) & (p.restore_reg_d <= T))
    exp = p.expire_reg_d.notna() & (p.expire_reg_d <= T)
    return p[(p.rg <= T) & ~lap & ~exp & (p.cndrtExptnDate_d > T)]
a22 = alive(2022); a24 = alive(2024)
res['n8_pop_t0_2022_approx'] = int(len(a22)); res['n8_pop_t0_2024_approx'] = int(len(a24))
ev = pd.read_sql("""select rgst_no, rgstCsDate d from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록')""", rc)
rc.close()
ev['d'] = pd.to_datetime(ev.d, format='%Y%m%d', errors='coerce')
def window_events(pop, a, b):
    e = ev[(ev.d > a) & (ev.d <= b) & ev.rgst_no.isin(pop.rgst_no)]
    return e.drop_duplicates('rgst_no')
e22 = window_events(a22, pd.Timestamp('2022-12-31'), pd.Timestamp('2024-12-31'))
young22 = a22[(a22.rg >= '2021-11-01') & (a22.rg <= '2022-08-31')]
res['n8_official_pos_approx'] = int(len(e22))
res['n8_official_analog_young_cohort_size'] = int(len(young22))
res['n8_official_analog_young_pos'] = int(e22.rgst_no.isin(young22.rgst_no).sum())
res['n8_official_analog_young_rate'] = float(e22.rgst_no.isin(young22.rgst_no).sum() / max(len(young22), 1))
res['n8_official_overall_rate'] = float(len(e22) / len(a22))
e24 = window_events(a24, pd.Timestamp('2024-12-31'), pd.Timestamp('2026-07-31'))
res['n8_short_pos_observed'] = int(len(e24))
res['n8_short_pop_rg_2023_11_to_2024_08_present'] = int(((a24.rg >= '2023-11-01') & (a24.rg <= '2024-08-31')).sum())
res['n8_short_pop_rg_2022_11_to_2023_08_present'] = int(((a24.rg >= '2022-11-01') & (a24.rg <= '2023-08-31')).sum())
json.dump(res, open(os.path.join(OUT, 's4_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))

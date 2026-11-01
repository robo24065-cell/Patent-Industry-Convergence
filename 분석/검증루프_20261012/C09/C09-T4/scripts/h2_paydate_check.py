# C09-T4 h2: 공휴일 목록의 실측 대조 — 등록원부 pay 납부일별 건수(전 행, seq 무관).
#  가설: 특허청 납부는 은행·전자납부 영업일에 처리되어 공휴일·주말 납부일 기록은 거의 없다.
#  (1) 목록 공휴일(평일)의 납부 건수가 주변 영업일 대비 거의 0인가  (2) 목록에 없는 평일 중 건수가 거의 0인 날(누락 후보)이 있는가
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
c = ro()
pdc = pd.read_sql("select payDate d, count(*) n from pay group by payDate", c)
pdc = pdc[pdc.d.str.fullmatch(r'\d{8}', na=False)]
pdc['dt'] = pd.to_datetime(pdc.d, format='%Y%m%d', errors='coerce')
pdc = pdc[pdc.dt.notna() & (pdc.dt >= '2000-01-01') & (pdc.dt <= '2026-07-31')]
cal = pd.DataFrame({'dt': pd.date_range('2000-01-01', '2026-07-31')})
cal = cal.merge(pdc[['dt', 'n']], on='dt', how='left').fillna({'n': 0})
hol = pd.read_csv(os.path.join(C, 'holidays_kr_2000_2026.csv'))
hol['dt'] = pd.to_datetime(hol.date); hol = hol.rename(columns={'name': 'hname'})
cal = cal.merge(hol[['dt', 'hname']], on='dt', how='left')
cal['wd'] = cal.dt.dt.weekday
cal['is_hol'] = cal.hname.notna()
cal['wkday'] = cal.wd < 5
# 기준: 같은 달 평일·비공휴일 중위 건수
cal['ym'] = cal.dt.dt.to_period('M')
med = cal[cal.wkday & ~cal.is_hol].groupby('ym').n.median()
cal['rel'] = cal.n / cal.ym.map(med)
cal['y'] = cal.dt.dt.year
res = {}
res['rel_weekend_median'] = float(cal[~cal.wkday].rel.median())
res['rel_weekend_p95'] = float(cal[~cal.wkday].rel.quantile(.95))
hw = cal[cal.wkday & cal.is_hol]
res['n_weekday_holidays'] = int(len(hw))
res['rel_weekday_holiday_median'] = float(hw.rel.median())
res['rel_weekday_holiday_p95'] = float(hw.rel.quantile(.95))
# 근로자의 날 따로
lab = hw[hw.hname.str.fullmatch('근로자의 날')]
res['labor_day_rel'] = {str(d.date()): round(float(r), 3) for d, r in zip(lab.dt, lab.rel)}
# 목록 공휴일(평일) 중 납부가 많은 날(rel > 0.3) — 목록 오류 후보
bad = hw[(hw.rel > 0.3) & ~hw.hname.str.fullmatch('근로자의 날')]
res['listed_weekday_holiday_but_busy'] = [(str(d.date()), n, round(float(r), 3)) for d, n, r in zip(bad.dt, bad.hname, bad.rel)]
# 목록 밖 평일 중 납부가 거의 없는 날(rel < 0.1) — 누락 후보
miss = cal[cal.wkday & ~cal.is_hol & (cal.rel < 0.1)]
res['unlisted_weekday_quiet'] = [(str(d.date()), int(n), round(float(r), 3)) for d, n, r in zip(miss.dt, miss.n, miss.rel)]
# 연도별: 평일 공휴일 상대 건수 중위, 주말 상대 건수 중위
res['by_year'] = cal.groupby('y').apply(lambda g: dict(wkend=round(float(g[~g.wkday].rel.median()), 4),
                                                         wkday_hol=round(float(g[g.wkday & g.is_hol & ~g.hname.fillna('').str.fullmatch('근로자의 날')].rel.median()), 4),
                                                         labor=round(float(g[g.hname.fillna('').str.fullmatch('근로자의 날')].rel.max()), 3) if (g.hname.fillna('') == '근로자의 날').any() else None,
                                                         n_quiet_unlisted=int((g.wkday & ~g.is_hol & (g.rel < 0.1)).sum()))).to_dict()
json.dump(res, open(os.path.join(C, 'h2_paydate_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
cal.drop(columns=['ym']).to_parquet(os.path.join(C, 'h2_paydate_calendar.parquet'), index=False)
print(json.dumps({k: v for k, v in res.items() if k != 'by_year'}, ensure_ascii=False, indent=1, default=str))
print(pd.DataFrame(res['by_year']).T.to_string())

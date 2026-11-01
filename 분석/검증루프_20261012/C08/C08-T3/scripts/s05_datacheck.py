# -*- coding: utf-8 -*-
"""데이터 검증 의무 (a)~(d) — 채점 정의는 바꾸지 않는다(사후 점검).
(a) 등록원부 vs KOSIS 지식재산통계(TX_13801_A006 등록, TX_13801_A052 존속권리) — 내려받은 json은 python -I 로 따로 읽어 csv로 둔 것을 사용
(b) 창 안 사건 분포·결측·중복  (c) 등록번호 공란 결손  (d) 스냅샷 시점·회복·창 끝 절단
+ 2위 대형 출원인(과대예측) 사건 유형 진단
출력: cache/s05_datacheck.json, cache/s05_*.csv"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import c08lib as C
L = C.L0; WEND = np.datetime64('2025-10-31'); SNAP = np.datetime64('2026-08-15')
D = C.load(); pt, pay, rt = D['pt'], D['pay'], D['rt']
fz = pd.read_parquet(os.path.join(C.W, 'frozen_scores.parquet'))
lb = pd.read_parquet(os.path.join(C.CACHE, 'labels.parquet'))
assert (lb.pid.values == fz.pid.values).all()
R = {}
lap = C.lapse_events(D)
# ================= (a) 공식 총량 대조
ko = pd.read_csv(os.path.join(C.CACHE, 's05_kosis.csv'))  # year, kipo_reg_patent, kipo_inforce_patent (s05a_kosis_read.py, python -I)
yrs = list(range(2010, 2026))
regy = pd.Series(pd.DatetimeIndex(pt.rd).year).value_counts()
pth_all, _ = C.paid_thru(pt, pay, SNAP)
ml = rt[rt.name.isin(C.MALSO)].groupby('pid').cd.min()
mld = np.full(len(pt), np.datetime64('NaT'), dtype='datetime64[ns]'); mld[ml.index.values] = ml.values
rows = []
for y_ in yrs:
    t = np.datetime64(f'{y_}-12-31')
    alive = (pt.rd.values <= t) & (pth_all > t) & (pt.exp.values > t) & ~(mld <= t)
    lap_E = int(((lap.E > np.datetime64(f'{y_ - 1}-12-31')) & (lap.E <= t)).sum())
    lap_cd = int(((lap.cd > np.datetime64(f'{y_ - 1}-12-31')) & (lap.cd <= t)).sum())
    rows.append(dict(year=y_, reg_registry=int(regy.get(y_, 0)), inforce_registry=int(alive.sum()), lapse_by_E=lap_E, lapse_by_cd=lap_cd))
a = pd.DataFrame(rows).merge(ko, on='year', how='left')
a['reg_share'] = a.reg_registry / a.kipo_reg_patent
a['inforce_share'] = a.inforce_registry / a.kipo_inforce_patent
a['kipo_exits'] = a.kipo_inforce_patent.shift(1) + a.kipo_reg_patent - a.kipo_inforce_patent
a['registry_exits'] = a.inforce_registry.shift(1) + a.reg_registry - a.inforce_registry
a['exit_share'] = a.registry_exits / a.kipo_exits
a['registry_lapse_rate_E'] = a.lapse_by_E / a.inforce_registry.shift(1)
a['kipo_exit_rate'] = a.kipo_exits / a.kipo_inforce_patent.shift(1)
a['registry_exit_rate'] = a.registry_exits / a.inforce_registry.shift(1)
a.to_csv(os.path.join(C.CACHE, 's05_a_official.csv'), index=False, encoding='utf-8-sig')
R['a'] = a.round(4).to_dict('records')
# ================= (b) 창 안 사건 분포
y = lb.y.values; due = fz.due_win.values == 1
port = fz.acd.map(fz[fz.acd != 'NA'].acd.value_counts()).fillna(1).values
pb = pd.cut(port, [0, 1, 9, 99, 999, 1e9], labels=['1', '2-9', '10-99', '100-999', '1000+']).astype(str)
ew = lap[(lap.E > L) & (lap.E <= WEND) & (lap.cd <= SNAP) & lap.pid.isin(fz.pid.values)].copy()
ew['Em'] = pd.DatetimeIndex(ew.E).strftime('%Y-%m')
R['b'] = dict(events_by_E_month=ew.Em.value_counts().sort_index().to_dict(),
              dup_right_rows_same_pid_cd=int(ew.duplicated(['pid', 'cd']).sum()),
              ksic_missing_riskset=int(fz.ksic.isna().sum() + (fz.ksic.astype(str).str.len() != 5).sum()),
              acd_NA_riskset=int((fz.acd == 'NA').sum()), acd_NA_pos=int(((fz.acd == 'NA').values & (y == 1)).sum()))
def dist(col):
    df = pd.DataFrame({'g': col, 'y': y, 'due': due})
    g = df.groupby('g').agg(n=('y', 'size'), due_n=('due', 'sum'), pos=('y', 'sum'))
    g['share_riskset'] = g.n / g.n.sum(); g['share_due'] = g.due_n / g.due_n.sum(); g['share_pos'] = g.pos / g.pos.sum()
    g['rate'] = g.pos / g.n; g['rate_due'] = g.pos / g.due_n.replace(0, np.nan)
    return g.reset_index()
db = []
for nm, col in (('atype', fz.atype.values), ('portfolio_size', pb), ('rgst_year', pd.DatetimeIndex(pt.rd.values[fz.pid.values]).year.astype(str)),
                ('ksic', fz.ksic.astype(str).values)):
    g = dist(col); g.insert(0, 'dim', nm); db.append(g)
db = pd.concat(db, ignore_index=True); db.to_csv(os.path.join(C.CACHE, 's05_b_dist.csv'), index=False, encoding='utf-8-sig')
R['b']['atype'] = db[db.dim == 'atype'].round(4).to_dict('records')
R['b']['portfolio_size'] = db[db.dim == 'portfolio_size'].round(4).to_dict('records')
kk = db[db.dim == 'ksic']
R['b']['ksic_rate_due_range'] = [float(kk.rate_due.min()), float(kk.rate_due.max())]
R['b']['ksic_max_share_pos'] = kk.sort_values('share_pos', ascending=False).head(5).round(4).to_dict('records')
# ================= (c) 등록번호 공란 결손
rdm = pd.Series(pd.DatetimeIndex(pt.rd).strftime('%Y-%m')).value_counts().sort_index()
R['c'] = dict(registry_reg_by_month_2022_2025={k: int(v) for k, v in rdm.items() if '2022-01' <= k <= '2025-08'})
coh = pd.DatetimeIndex(pt.rd.values[fz.pid.values])
for lo_, hi_ in (('2022-07-01', '2023-10-31'), ('2023-11-01', '2024-08-31'), ('2024-09-01', '2025-01-31'), ('2025-02-01', '2025-06-30')):
    m = (coh >= lo_) & (coh <= hi_)
    R['c'][f'riskset_rd_{lo_}_{hi_}'] = dict(n=int(m.sum()), due_win=int(due[m].sum()), pos=int(y[m].sum()))
p0 = pay[pay.seq == 0].drop_duplicates('pid').set_index('pid').l
rd_all = pd.DatetimeIndex(pt.rd)
m2 = (rd_all >= '2021-01-01') & (rd_all <= '2023-10-31')
R['c']['init_paid_years_rd2021_2023_10'] = pd.Series(pt.pid.values[m2]).map(p0).value_counts(normalize=True).round(4).to_dict()
# 공식 등록 대비 결손 추정 (2019–23 평균 몫 × KIPO)
sh = a[(a.year >= 2019) & (a.year <= 2023)].reg_share.mean()
R['c']['reg_share_2019_23_mean'] = float(sh)
R['c']['est_missing_2024'] = float(sh * a.set_index('year').kipo_reg_patent[2024] - a.set_index('year').reg_registry[2024])
R['c']['est_missing_2025'] = float(sh * a.set_index('year').kipo_reg_patent[2025] - a.set_index('year').reg_registry[2025])
# ================= (d) 스냅샷·회복·절단
rec = rt[rt.name == '회복등록']
ew['rec_cd'] = ew.pid.map(rec.groupby('pid').cd.max())
ewr = ew[ew.rec_cd.notna() & (ew.rec_cd >= ew.cd)]
ew['lag'] = (ew.cd - ew.E).dt.days
R['d'] = dict(window_events=len(ew), restored=len(ewr), restored_share=float(len(ewr) / len(ew)),
              restore_lag_days_q=ewr.assign(l=(ewr.rec_cd - ewr.cd).dt.days).l.quantile([0.5, 0.9, 0.99, 1.0]).to_dict(),
              lag_q=ew.lag.quantile([0.01, 0.5, 0.9, 0.99, 0.999, 1.0]).to_dict(),
              lag_by_E_month={m: dict(n=int(len(g)), p50=float(g.lag.median()), share_gt_287=float((g.lag > 287).mean()), max=int(g.lag.max()))
                              for m, g in ew.groupby('Em')},
              cd_max=str(ew.cd.max()), last2w=dict(n=int((ew.E > np.datetime64('2025-10-17')).sum()),
                                                    lag_max=int(ew[ew.E > np.datetime64('2025-10-17')].lag.max())))
# 창 끝 절단 추정: 역사적 287일 초과 비율(0.08–0.31%) × 마지막 사건 → 놓친 사건 상한
R['d']['est_missing_late_registrations_upper'] = float(0.0031 * len(ew))
# 양성 특허의 소멸등록 뒤 납부(사후 정정 가능성)
pos_p = ew[~ew.index.isin(ewr.index)][['pid', 'cd']]
pp = pay.merge(pos_p, on='pid'); R['d']['pos_with_pay_after_cd'] = int((pp.pd > pp.cd).groupby(pp.pid).any().sum())
# 창 안 기한 특허 중 2026-08-14까지 납부 미확인인데 소멸등록도 없는 특허(판정 미결)
pthS, _ = C.paid_thru(pt, pay, SNAP)
pidd = fz.pid.values[due]
undet = (pthS[pidd] <= WEND) & ~np.isin(pidd, ew.pid.values) & (pt.exp.values[pidd] > pthS[pidd])
ml_any = rt[rt.name.isin(C.MALSO) & (rt.cd > L)].pid.unique()
R['d']['due_unpaid_no_lapse_record'] = dict(n=int(undet.sum()), of_which_malso_after_L=int(np.isin(pidd[undet], ml_any).sum()))
# ================= 2위 출원인 진단
b2 = fz.pid.values[fz.big2.values == 1]
b2d = fz.pid.values[(fz.big2.values == 1) & due]
ev2 = rt[rt.pid.isin(b2) & (rt.cd > L)]
R['big2'] = dict(n=len(b2), due=len(b2d), events_after_L=ev2.groupby(['name', 'reason']).size().sort_values(ascending=False).head(12).reset_index().values.tolist(),
                 due_paid_beyond_window=int((pthS[b2d] > WEND).sum()), due_unpaid=int((pthS[b2d] <= WEND).sum()),
                 due_malso_after_L=int(np.isin(b2d, rt[rt.name.isin(C.MALSO) & (rt.cd > L)].pid.unique()).sum()))
# 2위 출원인의 과거 같은 창 실현율(동결 정보) — 지속 지수에 반영됐는지
Lp = np.datetime64('2024-07-01'); Wp = np.datetime64('2024-10-31')
rp, pthp, _ = C.risk_set(D, Lp)
acd_all = fz.set_index('pid').acd
ap1 = pd.read_parquet(os.path.join(C.CACHE, 'reg_pt.parquet'), columns=['pid', 'acd']).set_index('pid').acd
big2acd = fz.acd[fz.big2 == 1].iloc[0]
rp2 = rp[ap1.reindex(rp).values == big2acd]
e_prev = lap[(lap.E > Lp) & (lap.E <= Wp) & (lap.cd <= np.datetime64('2025-08-22'))].pid.unique()
R['big2']['prev_window_riskset'] = len(rp2); R['big2']['prev_window_events'] = int(np.isin(rp2, e_prev).sum())
for yy in (2021, 2022, 2023, 2024):
    Ly = np.datetime64(f'{yy}-07-01'); Wy = np.datetime64(f'{yy}-10-31')
    r_, _, _ = C.risk_set(D, Ly); r2 = r_[ap1.reindex(r_).values == big2acd]
    ee = lap[(lap.E > Ly) & (lap.E <= Wy)].pid.unique()
    R['big2'][f'window_{yy}'] = dict(riskset=len(r2), events=int(np.isin(r2, ee).sum()))
C.jdump(R, os.path.join(C.CACHE, 's05_datacheck.json'))
print(json.dumps(R, ensure_ascii=False, indent=1, default=str)[:20000])

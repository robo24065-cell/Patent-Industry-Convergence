# C09-T4 s1: 납부 행 단위 기한 요일 잡음 규모. 원자료 registry_full.sqlite(mode=ro).
#  대상: 특허(등록번호 10…) pay seq>0 (T4 정의) — statAnnl 결측이면 lastAnnl(T4와 같음). due = 등록일 + (s−1)년(2/29→2/28).
#  산출: cache/pay_ext.parquet(이후 단계 공용), noise_scale.csv(연도·권리자 규모·KSIC·기한 요일별)
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
c = ro()
pay = pd.read_sql("select rgst_no, seq, statAnnl, lastAnnl, payDate from pay where seq > 0", c)
pat = pd.read_sql("select rgst_no, rgstDate, ksic from patent", c)
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant", c)
a1 = ap.sort_values(['rgst_no', 'seq']).drop_duplicates('rgst_no').set_index('rgst_no').applicantCd
pay = pay[pay.rgst_no.str[:2] == '10']
pay['s'] = pd.to_numeric(pay.statAnnl, errors='coerce').fillna(pd.to_numeric(pay.lastAnnl, errors='coerce'))
pay['pd'] = pd.to_datetime(pay.payDate, format='%Y%m%d', errors='coerce').values.astype('datetime64[D]')
pay = pay.merge(pat, on='rgst_no', how='inner')
pay['rd'] = pd.to_datetime(pay.rgstDate, format='%Y%m%d', errors='coerce')
pay = pay[pay.s.notna() & pay.rd.notna() & ~np.isnat(pay.pd.values)].copy()
y = pay.rd.dt.year.values + pay.s.values.astype(int) - 1
m = pay.rd.dt.month.values; d = pay.rd.dt.day.values
d = np.where((m == 2) & (d == 29), 28, d)
pay['due'] = pd.to_datetime(pd.DataFrame({'year': y, 'month': m, 'day': d}), errors='coerce').values.astype('datetime64[D]')
pay = pay[~np.isnat(pay.due.values)].copy()
hol = load_hol()
due = pay.due.values.astype('datetime64[D]'); pdd = pay.pd.values.astype('datetime64[D]')
pay['dueR1'] = adjust_due(due, 'R1'); pay['dueR2'] = adjust_due(due, 'R2', hol)
pay['lag'] = (pdd - due).astype(int)
pay['lagR1'] = (pdd - pay.dueR1.values.astype('datetime64[D]')).astype(int)
pay['lagR2'] = (pdd - pay.dueR2.values.astype('datetime64[D]')).astype(int)
pay['due_dow'] = dow(due)
pay['pay_dow'] = dow(pdd)
pay['due_hol'] = np.isin(due, hol)
pay['acd'] = pay.rgst_no.map(a1).fillna('NA')
pay['py'] = pd.DatetimeIndex(pay.pd).year
pay['s'] = pay.s.astype(int)
pay = pay[['rgst_no', 'seq', 's', 'pd', 'due', 'dueR1', 'dueR2', 'lag', 'lagR1', 'lagR2', 'due_dow', 'pay_dow', 'due_hol', 'acd', 'ksic', 'py']]
pay.to_parquet(os.path.join(C, 'pay_ext.parquet'), index=False)
# ---- 분류
L0 = pay.lag > 0; L1 = pay.lagR1 > 0; L2 = pay.lagR2 > 0; L3 = pay.lag > 3
pay['wkd_due'] = pay.due_dow >= 5
pay['off_due'] = pay.wkd_due | pay.due_hol
pay['chg_R1'] = (L0 != L1); pay['chg_R2'] = (L0 != L2); pay['chg_R3'] = (L0 != L3)
pay['late0'] = L0; pay['lateR1'] = L1; pay['lateR2'] = L2; pay['lateR3'] = L3
top20 = pay[pay.acd != 'NA'].acd.value_counts().head(20).index
pay['size'] = np.where(pay.acd.isin(top20), 'top20', 'rest')
pay['period'] = np.where(pay.py < 2006, '2000-05', np.where(pay.py < 2015, '2006-14', np.where(pay.py < 2023, '2015-22', '2023-26')))
agg = dict(n=('lag', 'size'), weekend_due_share=('wkd_due', 'mean'), offday_due_share=('off_due', 'mean'),
           late_orig=('late0', 'mean'), late_R1=('lateR1', 'mean'), late_R2=('lateR2', 'mean'), late_R3=('lateR3', 'mean'),
           changed_R1=('chg_R1', 'mean'), changed_R2=('chg_R2', 'mean'), changed_R3=('chg_R3', 'mean'))
parts = []
for dim in ['py', 'size', 'ksic', 'due_dow', 'period']:
    g = pay.groupby(dim).agg(**agg).reset_index().rename(columns={dim: 'level'})
    g.insert(0, 'dim', dim); parts.append(g)
tot = pd.DataFrame([{k: (len(pay) if k == 'n' else pay[v[0]].mean()) for k, v in agg.items()}]); tot.insert(0, 'level', 'ALL'); tot.insert(0, 'dim', 'all')
# 기한이 주말·공휴일인 행 중 바뀐 비율(조건부)
cond = pay[pay.off_due]
tc = pd.DataFrame([{**{k: (len(cond) if k == 'n' else cond[v[0]].mean()) for k, v in agg.items()}}]); tc.insert(0, 'level', 'offday_due_only'); tc.insert(0, 'dim', 'cond')
# 기간 × 기한 요일(토요일 효과의 연혁 점검)
pp = pay.groupby(['period', 'due_dow']).agg(**agg).reset_index()
pp['level'] = pp.period + '_dow' + pp.due_dow.astype(str); pp = pp.drop(columns=['period', 'due_dow']); pp.insert(0, 'dim', 'period_x_due_dow')
# 규모 × 연도
sz = pay.groupby(['size', 'period']).agg(**agg).reset_index()
sz['level'] = sz['size'] + '_' + sz.period; sz = sz.drop(columns=['size', 'period']); sz.insert(0, 'dim', 'size_x_period')
ns = pd.concat([tot, tc] + parts + [pp, sz], ignore_index=True)
ns.to_csv(os.path.join(W, 'noise_scale.csv'), index=False, encoding='utf-8-sig')
# 지연 분포 세부: 주말 기한 지연의 lag 값
info = {}
for dw in range(7):
    s_ = pay[(pay.due_dow == dw) & (pay.lag > 0) & (pay.lag <= 7)].lag.value_counts().sort_index()
    info[f'lag1_7_due_dow{dw}'] = s_.to_dict()
info['pay_dow_share'] = pay.pay_dow.value_counts(normalize=True).sort_index().to_dict()
info['n_rows'] = int(len(pay)); info['py_range'] = [int(pay.py.min()), int(pay.py.max())]
info['top20_share_rows'] = float((pay['size'] == 'top20').mean())
info['ksic_n'] = int(pay.ksic.nunique()); info['ksic_na'] = float(pay.ksic.isna().mean())
info['dup_rgst_seq'] = int(pay.duplicated(['rgst_no', 'seq']).sum())
info['changed_R1_but_not_weekend'] = int((pay.chg_R1 & ~pay.wkd_due).sum())
info['R2_minus_R1_extra_changed'] = int((pay.chg_R2 & ~pay.chg_R1).sum())
json.dump(info, open(os.path.join(C, 's1_noise.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(info, ensure_ascii=False, indent=1, default=str)[:3000])
print(ns[ns.dim.isin(['all', 'cond', 'size', 'period', 'due_dow'])].round(4).to_string())

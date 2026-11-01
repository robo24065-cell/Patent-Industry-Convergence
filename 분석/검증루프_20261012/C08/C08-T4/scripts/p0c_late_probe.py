# C08-T4 P0-c (PREREG 전 정의 점검): 후보 E 지연일 분포, 요일 효과, 추가료 흔적. 캐시(사건 < 2025-01-01)만 사용.
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from lib8 import *
pat = pd.read_parquet(os.path.join(C, 'pat.parquet'))
pay = pd.read_parquet(os.path.join(C, 'pay.parquet'))
assert (pay.d.dropna() < CUT).all()
pay['d'] = ok8(pay.d)
for c in ['statAnnl', 'lastAnnl', 'payAmount']:
    pay[c] = pd.to_numeric(pay[c], errors='coerce')
p4 = pay[(pay.statAnnl >= 4) & pay.d.notna()].copy()
rd = p4.rgst_no.map(pat.set_index('rgst_no').rgstDate)
p4 = p4[rd.notna()].copy(); rd = rd[rd.notna()]
due_s = (rd.str[:4].astype(int) + p4.statAnnl.astype(int) - 1).astype(str) + rd.str[4:]
due_s = due_s.where(~due_s.str.endswith('0229'), due_s.str[:4] + '0228')
p4['due'] = pd.to_datetime(due_s, format='%Y%m%d', errors='coerce')
p4['dt'] = pd.to_datetime(p4.d, format='%Y%m%d')
p4 = p4[p4.due.notna()]
p4['lag'] = (p4.dt - p4.due).dt.days
p4['y'] = p4.dt.dt.year
dv = p4[(p4.y >= 2015) & (p4.y <= 2022)]
out = {}
out['n_rows_dev'] = int(len(dv))
out['lag_q'] = dv.lag.quantile([.01, .05, .25, .5, .75, .95, .99]).to_dict()
bins = [-10**6, -366, -31, -1, 0, 1, 2, 3, 5, 7, 14, 31, 92, 183, 214, 400, 10**6]
out["lag_bins"] = {str(k): int(v) for k, v in dv.lag.groupby(pd.cut(dv.lag, bins), observed=False).size().items()}
late = dv[dv.lag > 0]
out['late_lag1_7_by_due_weekday'] = late[late.lag <= 7].groupby(late.due.dt.dayofweek).size().astype(int).to_dict()
out['late_share_by_due_weekday'] = dv.groupby(dv.due.dt.dayofweek).lag.apply(lambda s: float((s > 0).mean())).to_dict()
out['late_gt3_share_by_due_weekday'] = dv.groupby(dv.due.dt.dayofweek).lag.apply(lambda s: float((s > 3).mean())).to_dict()
# 추가료 흔적: 같은 (납부연도, statAnnl, lastAnnl) 중위 금액 대비
g = dv[dv.payAmount > 0].copy()
g['rel'] = g.payAmount / g.groupby(['y', 'statAnnl', 'lastAnnl']).payAmount.transform('median')
for nm, m in {'ontime(lag<=0)': g.lag <= 0, 'lag1_3': (g.lag >= 1) & (g.lag <= 3), 'lag4_30': (g.lag >= 4) & (g.lag <= 30), 'lag31_183': (g.lag >= 31) & (g.lag <= 183), 'lag184_214': (g.lag >= 184) & (g.lag <= 214), 'lag>214': g.lag > 214}.items():
    out[f'rel_median_{nm}'] = float(g.rel[m].median()) if m.any() else None
    out[f'n_{nm}'] = int(m.sum())
out['zero_amount_share'] = float((dv.payAmount == 0).mean())
out['dup_rows(rgst_no,statAnnl)'] = int(dv.duplicated(['rgst_no', 'statAnnl']).sum())
json.dump(out, open(os.path.join(C, 'p0c_late_probe.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

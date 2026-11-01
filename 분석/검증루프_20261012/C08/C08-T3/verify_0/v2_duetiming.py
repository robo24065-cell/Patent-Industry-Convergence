# -*- coding: utf-8 -*-
"""verify_0/v2 — '같은 납부기한끼리' 주장 반박 시도: due_win(4개월) 안에서도 납부기한 월별 양성률이 31%→10%로 크게 다름.
납부기한 월·주·일 층 안 AUC(층별 쌍 가중 평균)로 skill 재계산 + 군집 부트스트랩 B=500. 출력 v2_out.json"""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
fz = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet'))
lb = pd.read_parquet(os.path.join(W, 'cache', 'labels.parquet'))
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
P = pd.read_sql("select rgst_no, rgstDate from patent order by rowid", con)
P['rd'] = pd.to_datetime(P.rgstDate, format='%Y%m%d', errors='coerce')
P = P[P.rd.notna() & P.rgst_no.str.startswith('10')].reset_index(drop=True)
rs = P.iloc[fz.pid.values].reset_index(drop=True)
d = np.where(fz.due_win.values == 1)[0]
rs = rs.iloc[d].reset_index(drop=True)
pay = pd.read_sql("select rgst_no, payDate, lastAnnl from pay", con)
pay = pay[pay.rgst_no.isin(set(rs.rgst_no))]
pay['pd'] = pd.to_datetime(pay.payDate, format='%Y%m%d', errors='coerce'); pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
lL = pay[pay.pd <= '2025-07-01'].groupby('rgst_no').l.max()
k = rs.rgst_no.map(lL).values.astype(int)
due = pd.DatetimeIndex([pd.Timestamp(t.year + kk, t.month, 28 if (t.month == 2 and t.day == 29) else t.day) for t, kk in zip(rs.rd, k)])
y = lb.y.values[d].astype(float); p = fz.p.values[d]; pb = fz.p_base.values[d]; ps = fz.p_strong.values[d]
acd = fz.acd.astype(str).values[d]; cl = np.where(acd == 'NA', 'P' + fz.pid.astype(str).values[d], acd)
codes, uniq = pd.factorize(cl); K = len(uniq)
day = (due - pd.Timestamp('2025-07-01')).days.values
strata = dict(month=due.month.values, half=(due.month.values * 2 + (due.day.values > 15)), week=day // 7, day=day)
def wauc_strat(s, st, yy, w):
    # 층 안 AUC를 층별 (W+ x W-) 쌍 가중 합산 = 층 안 비교만 하는 AUC
    key = pd.factorize(pd.Series(st).astype(str) + '|' + pd.Series(s).astype(str))[0]
    df = pd.DataFrame({'st': st, 's': s, 'wp': w * yy, 'wn': w * (1 - yy)})
    g = df.groupby(['st', 's'], sort=True)[['wp', 'wn']].sum().reset_index()
    g['cn'] = g.groupby('st').wn.cumsum() - g.wn
    num = (g.wp * (g.cn + 0.5 * g.wn)).sum()
    tot = df.groupby('st')[['wp', 'wn']].sum(); den = (tot.wp * tot.wn).sum()
    return float(num / den)
sk = lambda a, b: (a - b) / (1 - b)
O = {'n': int(len(d)), 'pos': int(y.sum())}
mon = pd.Series(y).groupby(due.month.values).agg(['sum', 'size', 'mean'])
O['by_month'] = {int(i): [int(r['sum']), int(r['size']), round(float(r['mean']), 4)] for i, r in mon.iterrows()}
wk = pd.Series(y).groupby(day // 7).mean()
O['rate_by_week'] = {int(i): round(float(v), 4) for i, v in wk.items()}
w1 = np.ones(len(y))
for nm, st in strata.items():
    a, ab, aS = wauc_strat(p, st, y, w1), wauc_strat(pb, st, y, w1), wauc_strat(ps, st, y, w1)
    O['strat_' + nm] = dict(auc=a, auc_base=ab, auc_strong=aS, skill=sk(a, ab), skill_strong=sk(a, aS))
O['nostrat_check'] = wauc_strat(p, np.zeros(len(y)), y, w1)
# 부트스트랩(월·일 층) B=300
rng = np.random.default_rng(99); R = []
for b in range(300):
    w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
    r = []
    for nm in ('month', 'day'):
        st = strata[nm]
        a, ab, aS = wauc_strat(p, st, y, w), wauc_strat(pb, st, y, w), wauc_strat(ps, st, y, w)
        r += [sk(a, ab), sk(a, aS)]
    R.append(r)
R = np.array(R)
O['boot300'] = {n: [float(np.percentile(R[:, i], 2.5)), float(np.percentile(R[:, i], 97.5))]
                for i, n in enumerate(['month_skill', 'month_skill_strong', 'day_skill', 'day_skill_strong'])}
O['secs'] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(HERE, 'v2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))

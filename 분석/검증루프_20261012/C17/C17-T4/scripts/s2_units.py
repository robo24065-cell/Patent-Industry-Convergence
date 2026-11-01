# C17-T4 s2: 단위(appl8 × 단계 2·3) 표. 개발 구역 이벤트(ev_dev_20250303)만 읽는다. LOCKED 폴더 금지.
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib17 import *
import numpy as np, pandas as pd, duckdb
t_start = time.time()
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
EV = (C + r'\ev_dev_20250303.parquet').replace('\\', '/')
assert 'LOCKED' not in EV
ev = con.execute(f"select distinct pat, appl8, filing, G, e, code from '{EV}' where regexp_matches(pat, '^[0-9]+$')").fetchdf()
n_raw = con.execute(f"select count(*) from '{EV}' where regexp_matches(pat, '^[0-9]+$')").fetchone()[0]
log = {'ev_numeric_rows_raw': int(n_raw), 'ev_numeric_rows_distinct': int(len(ev))}
# appl8 당 숫자 특허 1개만
pa_ = ev.groupby('appl8').pat.nunique()
amb = set(pa_[pa_ > 1].index)
log['appl8_multi_pat'] = len(amb)
ev = ev[~ev.appl8.isin(amb)]
pt = ev.groupby('pat').agg(appl8=('appl8', 'first'), G=('G', 'first'), filing=('filing', 'first'), nG=('G', 'nunique')).reset_index()
log['pat_multi_G'] = int((pt.nG > 1).sum())
pt = pt[pt.nG == 1].drop(columns='nG')
ev = ev[ev.pat.isin(pt.pat)]
log['n_pat'] = len(pt)

ev['pay_s'] = ev.code.map(lambda c: PAY[c][0] if c in PAY else 0)
ev['pay_e'] = ev.code.map(lambda c: PAY[c][1] if c in PAY else '')
ev['is_exp'] = ev.code == 'EXP.'
ev['is_expx'] = ev.code == 'EXPX'
ev['is_pet'] = ev.code.isin(['PMFP', 'PMFG', 'PMFD', 'PMFS'])
ev['is_sg'] = ev.code.isin(SURCH_GRACE)
ev['is_stat'] = ev.code.isin(['LTOS', 'SMAL', 'BIG.', 'MICR', 'MTOS', 'STOL', 'LSM1', 'LSM2', 'LSM3'])

def add_months_vec(D, k):
    return pd.Series([add_months(int(d), k) for d in D], index=D.index)


rows = []
for s in (2, 3):
    u = pt.copy()
    u['s'] = s
    u['D'] = [add_years(g, 4 * s) for g in u.G]
    u['t0'] = [add_months(d, -12) for d in u.D]
    u['Dprev'] = [add_years(g, 4 * (s - 1)) for g in u.G]
    u = u[(u.D >= 20160101) & (u.D <= 20251231)]
    # 출원일 + 20년 ≤ D 제외
    u = u[[add_years(f, 20) > d for f, d in zip(u.filing, u.D)]]
    m = ev.merge(u[['pat', 'D', 't0', 'Dprev']], on='pat')
    pre = m[m.e < m.t0]
    # 직전 단계 납부(원점 이전)
    pp = pre[pre.pay_s == s - 1].sort_values('e').groupby('pat').agg(pay_prev_e=('e', 'last'), ent_prev=('pay_e', 'last'))
    u = u.merge(pp, on='pat', how='inner')
    # 직전 단계 유예 가산금(직전 마감 ±7개월)
    sg = pre[pre.is_sg].merge(u[['pat']], on='pat')
    sg = sg[(sg.e >= [add_months(d, -7) for d in sg.Dprev]) & (sg.e <= [add_months(d, 7) for d in sg.Dprev])]
    u['late_prev'] = u.pat.isin(set(sg.pat)).astype(int)
    for col, flag in [('n_pet', 'is_pet'), ('n_expx', 'is_expx'), ('n_stat', 'is_stat'), ('n_exp_prior', 'is_exp')]:
        c = pre[pre[flag]].groupby('pat').size()
        u[col] = u.pat.map(c).fillna(0).astype(int)
    # 직전 납부 시점: 직전 마감 대비 일수
    u['pay_prev_lead'] = [(ymd2date(d) - ymd2date(p)).days for d, p in zip(u.Dprev, u.pay_prev_e)]
    # 결과(개발만): EXP. 입력일 ∈ [D−7, D+45]
    dev = u.D < LOCK_START
    ex = m[m.is_exp | m.is_expx][['pat', 'e', 'D']]   # 변경 1: 회복 시 EXP.가 EXPX(입력일=D)로 바뀌므로 둘 다 소멸로 본다
    ex = ex[[(ymd2date(e) - ymd2date(d)).days >= -7 and (ymd2date(e) - ymd2date(d)).days <= 45 for e, d in zip(ex.e, ex.D)]]
    u['y'] = np.where(dev, u.pat.isin(set(ex.pat)).astype(float), np.nan)
    exx = m[m.is_expx & (m.e >= add_months_vec(m.D, -1)) & (m.e <= add_months_vec(m.D, 2))].pat
    u['expx_after'] = np.where(dev, u.pat.isin(set(exx)).astype(float), np.nan)
    # 같은 단계 납부(개발만; 검산용)
    paid = m[(m.pay_s == s) & (m.e >= m.t0)].pat
    u['paid_s'] = np.where(dev, u.pat.isin(set(paid)).astype(float), np.nan)
    rows.append(u)
U = pd.concat(rows, ignore_index=True)
# 제공 문서 결합
docs = pd.read_parquet(C + r'\us_docs.parquet', columns=['appl8', 'appl_date', 'appl_year', 'rgst_no', 'rgst_date', 'applicant', 'main_ipc', 'family', 'prio_cc', 'ksic'])
U = U.merge(docs, on='appl8', how='left')
assert U.ksic.notna().all()
U.to_parquet(C + r'\units.parquet', index=False)
d = U[U.D < LOCK_START]
log['units_total'] = len(U)
log['units_by_s_Dyear'] = {f'{s}|{y}': int(n) for (s, y), n in U.groupby(['s', U.D // 10000]).size().items()}
log['dev_rate_by_s_Dyear'] = {f'{s}|{y}': round(float(r), 4) for (s, y), r in d.groupby(['s', d.D // 10000]).y.mean().items()}
log['dev_y_and_paid_both'] = int(((d.y == 1) & (d.paid_s == 1)).sum())
log['dev_neither'] = int(((d.y == 0) & (d.paid_s == 0)).sum())
log['sec'] = round(time.time() - t_start, 1)
json.dump(log, open(C + r'\s2_log.json', 'w'), indent=1)
out(json.dumps(log, indent=1))

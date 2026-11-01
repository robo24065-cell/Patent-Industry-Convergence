# -*- coding: utf-8 -*-
"""verify_2 반박 검증 — C08-T3 보정 주장(1.11배, 10분위 0.69~5.73, 위험모형 창 합 0.86배) 재계산.
c08lib·s04 코드 미사용. 레이블은 sqlite(mode=ro) 직접 SQL로 새로 생성(E = 등록일 + 소멸등록 전 최대 납부연차).
추가(사후 분석, 판정용): 10분위 경사가 '분산 좁음' 때문인지 '창 안 납부기한 보유 비율' 차이 때문인지 분해.
출력: verify_2/v2_out.json"""
import os, sys, json, sqlite3, hashlib, time, datetime as dt
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
O = {}
fp = os.path.join(W, 'frozen_scores.parquet')
O['frozen_sha256'] = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
fz = pd.read_parquet(fp)
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
P = pd.read_sql("select rowid rid, rgst_no, rgstDate from patent order by rowid", con)
P = P[P.rgstDate.str.match(r'^\d{8}$', na=False) & P.rgst_no.str.startswith('10')].reset_index(drop=True)
rs = P.iloc[fz.pid.values].reset_index(drop=True)
rset = set(rs.rgst_no)
# 불납 소멸등록
lap = pd.read_sql("""select rgst_no, rgstCsDate cd, rgstCsReason r from right where rgstCsName='소멸등록'""", con)
O['lapse_reasons_all'] = lap.r.value_counts().head(8).to_dict()
lap = lap[lap.r.isin(['등록료불납', '연차료불납']) & lap.cd.str.match(r'^\d{8}$', na=False) & (lap.cd <= '20260815')]
lap = lap[lap.rgst_no.isin(rset)].drop_duplicates(['rgst_no', 'cd']).reset_index(drop=True)
pay = pd.read_sql("select rgst_no, payDate, lastAnnl from pay", con)
pay = pay[pay.rgst_no.isin(set(lap.rgst_no)) & pay.payDate.str.match(r'^\d{8}$', na=False)]
pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
m = lap.reset_index().merge(pay[['rgst_no', 'payDate', 'l']], on='rgst_no')
m = m[m.payDate < m.cd]
lap['lmax'] = lap.index.map(m.groupby('index').l.max())
O['lapse_no_pay_before_cd'] = int(lap.lmax.isna().sum())
lap = lap[lap.lmax.notna()].copy()
rd = dict(zip(P.rgst_no, P.rgstDate))
def E(r, k):
    s = rd[r]; y, mo, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    if mo == 2 and d == 29: d = 28
    return dt.date(y + int(k), mo, d)
lap['E'] = [E(r, k) for r, k in zip(lap.rgst_no, lap.lmax)]
win = lap[(lap.E > dt.date(2025, 7, 1)) & (lap.E <= dt.date(2025, 10, 31))].copy()
rec = pd.read_sql("select rgst_no, rgstCsDate rcd from right where rgstCsName='회복등록' and rgstCsDate <= '20260815'", con)
rmax = rec.groupby('rgst_no').rcd.max()
win['rest'] = win.rgst_no.map(rmax).fillna('0') >= win.cd
y = rs.rgst_no.isin(set(win[~win.rest].rgst_no)).astype(int).values
yS1 = rs.rgst_no.isin(set(win.rgst_no)).astype(int).values
lb = pd.read_parquet(os.path.join(W, 'cache', 'labels.parquet'))
O['labels'] = dict(pos=int(y.sum()), pos_S1=int(yS1.sum()), mismatch_vs_s04=int((y != lb.y.values).sum()))
p = fz.p.values; phw = fz.p_haz_win.values; due = fz.due_win.values == 1
pw = 1 - (1 - p) ** (122 / 1096)
O['calib'] = dict(sum_pw=float(pw.sum()), ratio_pw=float(pw.sum() / y.sum()), ratio_pw_S1=float(pw.sum() / yS1.sum()),
                  sum_hazwin=float(phw.sum()), ratio_hazwin=float(phw.sum() / y.sum()), ratio_hazwin_S1=float(phw.sum() / yS1.sum()),
                  pos_outside_due=int(y[~due].sum()), due_n=int(due.sum()))
# 군집 부트스트랩(다른 구현: 군집 다항 가중, seed 2026)
cl = np.where(fz.acd.values == 'NA', 'P' + fz.pid.astype(str).values, fz.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)
spw = np.bincount(codes, pw, K); shw = np.bincount(codes, phw, K); sy = np.bincount(codes, y, K)
rng = np.random.default_rng(2026); r1, r2 = [], []
for _ in range(1000):
    w = rng.multinomial(K, np.full(K, 1 / K))
    r1.append((w * spw).sum() / (w * sy).sum()); r2.append((w * shw).sum() / (w * sy).sum())
O['calib']['ratio_pw_ci'] = [float(np.percentile(r1, 2.5)), float(np.percentile(r1, 97.5))]
O['calib']['ratio_hazwin_ci'] = [float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))]
# 10분위 (p 내림차순 1=최상위)
d = pd.DataFrame({'p': p, 'pw': pw, 'phw': phw, 'y': y, 'due': due.astype(int)})
d['dec'] = pd.qcut(-d.p.rank(method='first'), 10, labels=False) + 1
g = d.groupby('dec').agg(n=('y', 'size'), pw=('pw', 'sum'), phw=('phw', 'sum'), ev=('y', 'sum'), due=('due', 'mean'))
g['ratio_pw'] = g.pw / g.ev; g['ratio_hw'] = g.phw / g.ev
# 분해: 기한 보유 특허 안에서의 실현율, 그리고 '기한 보유 특허에 1년 환산 확률'을 준 시간 정합 환산
d['p1y'] = 1 - (1 - d.p) ** (365.25 / 1096)
dd = d[d.due == 1]
g2 = dd.groupby('dec').agg(n_due=('y', 'size'), ev_due=('y', 'sum'), pw_due=('pw', 'sum'), p1y_due=('p1y', 'sum'))
g = g.join(g2)
g['realized_in_due'] = g.ev_due / g.n_due
g['ratio_pw_within_due'] = g.pw_due / g.ev_due
g['ratio_p1y_due'] = g.p1y_due / g.ev_due
O['deciles'] = g.reset_index().round(5).to_dict('records')
O['p1y_due_total_ratio'] = float(dd.p1y.sum() / dd.y.sum())
# due 안 10분위(due 자체로 다시 자름)
dd = dd.copy(); dd['dq'] = pd.qcut(-dd.p.rank(method='first'), 10, labels=False) + 1
g3 = dd.groupby('dq').agg(pw=('pw', 'sum'), p1y=('p1y', 'sum'), ev=('y', 'sum'))
O['due_deciles_ratio_pw'] = (g3.pw / g3.ev).round(4).tolist()
O['due_deciles_ratio_p1y'] = (g3.p1y / g3.ev).round(4).tolist()
# 노출 근거
cj = os.path.join(BASE, r"분석\검증루프_20261012\C01\C01-T4\verify_2\v2_cohort.json")
O['c01_cohort_exists'] = os.path.exists(cj)
if O['c01_cohort_exists']:
    O['c01_cohort_head'] = open(cj, encoding='utf-8').read()[:1500]
O['secs'] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(HERE, 'v2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))

# C09-T4 반박 검증(독립 재계산): verify_1 / v1b 의 '주말 기한 다음 영업일 납부 = 진짜 지연에 가깝다', '신호 정체 = 마지막 순간 납부'
#  - 캐시(t4_pay·loaded_r2dev.pkl) 대신 registry_full.sqlite(mode=ro)에서 직접 다시 만든다.
#  - 점검 가설: 결과창 (pd, pd+3y] 은 납부일 기준이라, 기한 전에 낸 납부(e>0)는 '3번째 다음 기념일 소멸(E=due+3y)'이 창 밖으로 빠지고
#    기한 당일·후 납부(e<=0)는 들어간다 → 집단 간 기계적 비대칭. 기한 기준 창 (pd, due+3y] 과 '다음 해 불납'(E=due+1y)으로 다시 비교.
import os, sys, json, sqlite3
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B + r'\수집파이프라인\registry_full.sqlite'
W = B + r'\분석\검증루프_20261012\C09\C09-T4'
c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
pay = pd.read_sql("select rgst_no, seq, statAnnl, lastAnnl, payDate from pay where seq > 0 and substr(rgst_no,1,2)='10'", c)
pat = pd.read_sql("select rgst_no, rgstDate from patent where substr(rgst_no,1,2)='10'", c)
rt = pd.read_sql("select rgst_no, rgstCsDate from right where rgstCsName='소멸등록' and rgstCsReason in ('등록료불납','연차료불납')", c)
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant where substr(rgst_no,1,2)='10'", c)
c.close()
d64 = lambda s: pd.to_datetime(pd.Series(s).astype(str).str.strip(), format='%Y%m%d', errors='coerce').values.astype('datetime64[D]')


def add_years(dates, k):
    di = pd.DatetimeIndex(dates); y = di.year.values + np.asarray(k); m = di.month.values; d = di.day.values
    d = np.where((m == 2) & (d == 29), 28, d)
    return pd.to_datetime(pd.DataFrame({'year': y, 'month': m, 'day': d}), errors='coerce').values.astype('datetime64[D]')


pat['rd'] = d64(pat.rgstDate); pat = pat[~np.isnat(pat.rd.values)]
rd = pat.set_index('rgst_no').rd
pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce'); pay['s'] = pd.to_numeric(pay.statAnnl, errors='coerce')
pay['pd'] = d64(pay.payDate)
pay = pay[pay.l.notna() & ~np.isnat(pay.pd.values) & pay.rgst_no.isin(rd.index)].copy()
pay['s'] = pay.s.fillna(pay.l).astype(int); pay['l'] = pay.l.astype(int)
pay['rd'] = pay.rgst_no.map(rd).values.astype('datetime64[D]')
pay['due'] = add_years(pay.rd.values, pay.s.values - 1)
pay = pay[~np.isnat(pay.due.values)].copy()
pay['e'] = (pay.due.values - pay.pd.values).astype('timedelta64[D]').astype(int)
info = {'n_pay_seq_gt0': int(len(pay))}
# 기한 검증: 평일 기한의 e 분포(-3..3)
dow = lambda d: (np.asarray(d, dtype='datetime64[D]').view('int64') + 3) % 7
wd = dow(pay.due.values) < 5
info['weekday_due_e_counts_-5..5'] = pay[wd & (pay.e.abs() <= 5)].e.value_counts().sort_index().to_dict()
# 소멸 사건
rt['cd'] = d64(rt.rgstCsDate); rt = rt[~np.isnat(rt.cd.values)].reset_index(drop=True); rt['k'] = np.arange(len(rt))
mm = rt.merge(pay[['rgst_no', 'pd', 'l']], on='rgst_no')
mm = mm[mm.pd.values < mm.cd.values]
lmax = mm.groupby('k').l.max()
rt['lmax'] = rt.k.map(lmax); rt = rt[rt.lmax.notna()].copy()
rt['E'] = add_years(rt.rgst_no.map(rd).values.astype('datetime64[D]'), rt.lmax.values.astype(int))
rt = rt[rt.E.values < np.datetime64('2022-01-01')][['rgst_no', 'E']]
info['n_lapse_E_lt2022'] = int(len(rt))
# 공휴일(C09 캐시 목록; 주말만 쓰는 R1 판도 병기)
hol = np.array(sorted(set(pd.to_datetime(pd.read_csv(os.path.join(W, 'cache', 'holidays_kr_2000_2026.csv')).date).values.astype('datetime64[D]'))))
p = pay[(pay.pd.values >= np.datetime64('2008-01-01')) & (pay.pd.values <= np.datetime64('2017-12-31'))].copy()
due = p.due.values.astype('datetime64[D]'); pdd = p.pd.values.astype('datetime64[D]')
a1 = ap.sort_values(['rgst_no', 'seq']).drop_duplicates('rgst_no').set_index('rgst_no').applicantCd
p['acd'] = p.rgst_no.map(a1).fillna('NA')
p['atype'] = p.acd.str[0].map({'1': 'corp', '2': 'univ', '3': 'inst', '4': 'indiv', '5': 'fcorp', '6': 'findiv'}).fillna('other')
p['sb'] = pd.cut(p.s, [0, 6, 9, 12, 99], labels=['4-6', '7-9', '10-12', '13+']).astype(str)
p['st'] = p.atype + '|' + p.sb
p = p.reset_index(drop=True); p['rid'] = np.arange(len(p))
end_pd = (pd.DatetimeIndex(pdd) + pd.DateOffset(years=3)).values.astype('datetime64[D]')
end_due3 = add_years(due, 3)
nxt = add_years(due, 1)
m2 = p[['rid', 'rgst_no']].merge(rt, on='rgst_no')
Ev = m2.E.values.astype('datetime64[D]'); r_ = m2.rid.values
pd_r = pdd[r_]
p['y_pd3'] = np.isin(p.rid, r_[(Ev > pd_r) & (Ev <= end_pd[r_])]).astype(float)       # 원 정의
p['y_due3'] = np.isin(p.rid, r_[(Ev > pd_r) & (Ev <= end_due3[r_])]).astype(float)    # 기한 기준 3기념일
p['y_next'] = np.isin(p.rid, r_[(Ev > pd_r) & (Ev <= nxt[r_])]).astype(float)         # 바로 다음 해 불납
p['y_due3_pdend'] = np.isin(p.rid, r_[(Ev > pd_r) & (Ev <= np.minimum(end_due3, end_pd)[r_])]).astype(float)  # 두 창 교집합(=모든 집단 2~3기회 중 짧은 쪽)
off2 = (dow(due) >= 5) | np.isin(due, hol)
dR2 = np.busday_offset(due, 0, roll='forward', weekmask='1111100', holidays=hol)
e = p.e.values; e2 = (dR2 - pdd).astype(int)
nb = np.busday_offset(due, 1, roll='forward', weekmask='1111100', holidays=hol)
G = {}
G['A'] = (~off2) & (e >= 4)
G['B'] = (~off2) & (e >= 0) & (e <= 3)
G['B0'] = (~off2) & (e == 0)
G['B13'] = (~off2) & (e >= 1) & (e <= 3)
G['W'] = off2 & (e < 0) & (e2 >= 0)
G['W1'] = off2 & (e < 0) & (e2 == 0)
G['D'] = (~off2) & (e <= -1) & (e >= -30)
G['D1'] = (~off2) & (pdd == nb)
G['D2'] = (~off2) & (e <= -2) & (e >= -30) & (pdd != nb)
G['W2'] = off2 & (e2 <= -1) & (e2 >= -30)
main = np.zeros(len(p), bool)
for k in ['A', 'B', 'W', 'D']: main |= G[k]
wst = p[main].st.value_counts(normalize=True)
Y = ['y_pd3', 'y_due3', 'y_next', 'y_due3_pdend']
cl = np.where(p.acd.values == 'NA', 'P' + p.rgst_no.values, p.acd.values)
codes, uniq = pd.factorize(cl)


def stdr(sub_idx, ycol, w=None):
    """sub_idx: 행 인덱스(중복 허용). 층별 평균을 wst로 직접 표준화."""
    yv = p[ycol].values[sub_idx]; st = p.st.values[sub_idx]
    df = pd.DataFrame({'y': yv, 'st': st})
    r = df.groupby('st').y.mean().reindex(wst.index).fillna(df.y.mean())
    return float((r.values * wst.values).sum())


out = dict(info=info, n={k: int(v.sum()) for k, v in G.items()})
idx = {k: np.where(v)[0] for k, v in G.items()}
for yc in Y:
    out[yc] = {'raw': {k: round(float(p[yc].values[ix].mean()), 5) for k, ix in idx.items()},
               'std': {k: round(stdr(ix, yc), 5) for k, ix in idx.items()}}
# 군집 부트스트랩(출원인코드, 결측은 특허) 200회 — 핵심 차이만
rng = np.random.default_rng(1)
K = len(uniq)
order = np.argsort(codes, kind='stable'); bounds = np.searchsorted(codes[order], np.arange(K + 1))
cnt = np.diff(bounds)
pairs = [('W', 'B'), ('W', 'D'), ('B0', 'B13'), ('W1', 'B0'), ('D1', 'B0'), ('D2', 'D1')]
bs = {yc: {f'{a}-{b}': [] for a, b in pairs} for yc in Y}
for it in range(200):
    w = np.bincount(rng.integers(0, K, K), minlength=K)   # 군집 반복 횟수
    rw = np.empty(len(p)); rw[order] = np.repeat(w, cnt)
    for yc in Y:
        yv = p[yc].values
        s_ = {}
        for k in set(sum(pairs, ())):
            ix = idx[k]; ww = rw[ix]
            df = pd.DataFrame({'yw': yv[ix] * ww, 'w': ww, 'st': p.st.values[ix]}).groupby('st')[['yw', 'w']].sum()
            r = (df.yw / df.w.replace(0, np.nan)).reindex(wst.index)
            r = r.fillna((yv[ix] * ww).sum() / max(ww.sum(), 1))
            s_[k] = float((r.values * wst.values).sum())
        for a, b in pairs: bs[yc][f'{a}-{b}'].append(s_[a] - s_[b])
for yc in Y:
    out[yc]['diff'] = {}
    for a, b in pairs:
        v = np.array(bs[yc][f'{a}-{b}'])
        out[yc]['diff'][f'{a}-{b}'] = [round(out[yc]['std'][a] - out[yc]['std'][b], 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    s = out[yc]['std']
    out[yc]['prereg_rule_W_closer_to_D'] = bool(abs(s['W'] - s['D']) < abs(s['W'] - s['B']))
json.dump(out, open(os.path.join(W, 'verify_1', 'v1r_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

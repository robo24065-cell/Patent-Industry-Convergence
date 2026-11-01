# C09-T4 verify_0 (반박 검증 2차, 2026-10-13): '주말·공휴일 기한 잡음' 발견을 원자료에서 독립 재계산.
#  c9lib·s1_noise 코드를 쓰지 않는다. sqlite mode=ro. 공휴일은 cache/holidays_kr_2000_2026.csv(검증된 목록)만 읽는다.
#  산출: verify_0/r1_out.json, verify_0/r1_groups.csv
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B + r'\수집파이프라인\registry_full.sqlite'
W = B + r'\분석\검증루프_20261012\C09\C09-T4'
O = W + r'\verify_0'
t = time.time()
c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
pay = pd.read_sql("select p.rgst_no, p.seq, p.statAnnl, p.lastAnnl, p.payDate, t.rgstDate, t.ksic from pay p join patent t on p.rgst_no=t.rgst_no "
                  "where p.seq > 0 and substr(p.rgst_no,1,2)='10'", c)
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant where substr(rgst_no,1,2)='10'", c)
print('load', len(pay), f'{time.time()-t:.0f}s', flush=True)
out = {}
out['n_raw'] = int(len(pay))
s = pd.to_numeric(pay.statAnnl, errors='coerce')
s = s.where(s.notna(), pd.to_numeric(pay.lastAnnl, errors='coerce'))
out['n_statAnnl_na'] = int(pd.to_numeric(pay.statAnnl, errors='coerce').isna().sum())
rd = pd.to_datetime(pay.rgstDate, format='%Y%m%d', errors='coerce')
pdt = pd.to_datetime(pay.payDate, format='%Y%m%d', errors='coerce')
ok = s.notna() & rd.notna() & pdt.notna()
out['n_drop'] = int((~ok).sum())
pay = pay[ok].copy(); s = s[ok].astype(int); rd = rd[ok]; pdt = pdt[ok]
# 기한: 등록 기념일(s-1년 뒤), 2/29 -> 2/28  (DateOffset 대신 문자열로 직접 구성)
yy = rd.dt.year + s - 1; mm = rd.dt.month; dd = rd.dt.day.where(~((rd.dt.month == 2) & (rd.dt.day == 29)), 28)
due = pd.to_datetime(yy.astype(str) + '-' + mm.astype(str).str.zfill(2) + '-' + dd.astype(str).str.zfill(2), format='%Y-%m-%d')
due = due.values.astype('datetime64[D]'); pd_ = pdt.values.astype('datetime64[D]')
hol = pd.read_csv(os.path.join(W, 'cache', 'holidays_kr_2000_2026.csv'))
H = np.array(sorted(set(pd.to_datetime(hol.date).values.astype('datetime64[D]'))))
wd = pd.DatetimeIndex(due).weekday.values  # 월0..일6 (pandas 독립 계산)
dueR1 = due + np.where(wd == 5, 2, np.where(wd == 6, 1, 0)).astype('timedelta64[D]')
dueR2 = np.busday_offset(due, 0, roll='forward', weekmask='Mon Tue Wed Thu Fri', holidays=H)
lag = (pd_ - due).astype(int); l1 = (pd_ - dueR1).astype(int); l2 = (pd_ - dueR2).astype(int)
L0 = lag > 0; L1 = l1 > 0; L2 = l2 > 0; L3 = lag > 3
out['n_rows'] = int(len(lag))
out['s_eq2_rows'] = int((s == 2).sum()); out['s_lt4_rows'] = int((s < 4).sum())
out['late'] = {k: float(v.mean()) for k, v in dict(orig=L0, R1=L1, R2=L2, R3=L3).items()}
out['share_of_orig_late_removed'] = {k: float(1 - v.mean() / L0.mean()) for k, v in dict(R1=L1, R2=L2, R3=L3).items()}
out['late_statAnnl_ge4_only'] = {k: float(v[s.values >= 4].mean()) for k, v in dict(orig=L0, R1=L1, R2=L2, R3=L3).items()}
out['n_statAnnl_ge4'] = int((s >= 4).sum())
# 요일·lag 히스토그램
hist = {}
for w in range(7):
    m = wd == w
    vc = pd.Series(lag[m]).value_counts()
    hist[str(w)] = {str(k): int(vc.get(k, 0)) for k in range(-3, 8)}
out['lag_hist_by_due_dow_mon0'] = hist
# 그룹: 1순위 출원인(applicant seq 최소)
a1 = ap.sort_values(['rgst_no', 'seq']).drop_duplicates('rgst_no').set_index('rgst_no').applicantCd
acd = pay.rgst_no.map(a1).fillna('NA').values
vc = pd.Series(acd[acd != 'NA']).value_counts()
top20 = set(vc.index[:20])
size = np.where(np.isin(acd, list(top20)), 'top20', 'rest')
df = pd.DataFrame(dict(acd=acd, size=size, ksic=pay.ksic.values, py=pd.DatetimeIndex(pd_).year, wkd=wd >= 5,
                       off=(wd >= 5) | np.isin(due, H), L0=L0, L1=L1, L2=L2, L3=L3, chg1=L0 & ~L1, chg2=L0 & ~L2))
out['chg_reverse_R1'] = int((~L0 & L1).sum()); out['chg_reverse_R2'] = int((~L0 & L2).sum())
def agg(g):
    r = g.agg(n=('L0', 'size'), wkd=('wkd', 'mean'), off=('off', 'mean'), L0=('L0', 'mean'), L1=('L1', 'mean'), L2=('L2', 'mean'),
              L3=('L3', 'mean'), chg1=('chg1', 'mean'), chg2=('chg2', 'mean'))
    r['noise_frac_R1'] = r.chg1 / r.L0; r['noise_frac_R2'] = r.chg2 / r.L0
    return r
gs = []
for dim in ['size', 'ksic', 'py']:
    g = agg(df.groupby(dim)).reset_index().rename(columns={dim: 'level'}); g.insert(0, 'dim', dim); gs.append(g)
# 권리자 지연 성향(원판 L0 비율) 10분위: 납부 50건 이상 권리자, 권리자 단위(행 가중)
a = df[df.acd != 'NA'].groupby('acd').agg(n=('L0', 'size'), lr=('L0', 'mean'))
a = a[a.n >= 50]
a['bin'] = pd.qcut(a.lr.rank(method='first'), 10, labels=False)
df['abin'] = df.acd.map(a.bin)
g = agg(df[df.abin.notna()].groupby('abin')).reset_index().rename(columns={'abin': 'level'}); g.insert(0, 'dim', 'applicant_late_decile_n50'); gs.append(g)
# 주말 기한 조건부: 주말기한 행 중 바뀐 비율 vs 평일기한 R1 지연률(진짜 지연 성향)
for nm, sub in [('top20', df['size'] == 'top20'), ('rest', df['size'] == 'rest')]:
    x = df[sub]
    out[f'cond_{nm}'] = dict(chg1_given_wkd=float(x[x.wkd].chg1.mean()), late_weekday_due=float(x[~x.wkd].L0.mean()),
                             late_R1_wkd_due=float(x[x.wkd].L1.mean()))
G = pd.concat(gs, ignore_index=True)
G.to_csv(os.path.join(O, 'r1_groups.csv'), index=False, encoding='utf-8-sig')
ks = G[G.dim == 'ksic']; ksn = ks[ks.n >= 1000]
out['ksic'] = dict(n=int(len(ks)), wkd_min=float(ks.wkd.min()), wkd_max=float(ks.wkd.max()), wkd_sd=float(ks.wkd.std()),
                   chg1_min=float(ks.chg1.min()), chg1_max=float(ks.chg1.max()), L0_min=float(ks.L0.min()), L0_max=float(ks.L0.max()),
                   noise_frac_R1_min=float(ks.noise_frac_R1.min()), noise_frac_R1_max=float(ks.noise_frac_R1.max()),
                   noise_frac_R1_n1000_min=float(ksn.noise_frac_R1.min()), noise_frac_R1_n1000_max=float(ksn.noise_frac_R1.max()),
                   corr_L0_noisefrac=float(np.corrcoef(ks.L0, ks.noise_frac_R1)[0, 1]),
                   loglog_slope_chg1_on_L0=float(np.polyfit(np.log(ks.L0), np.log(ks.chg1), 1)[0]))
py = G[G.dim == 'py']
out['py'] = {str(int(r.level)): [int(r.n), round(float(r.wkd), 4)] for r in py.itertuples()}
dec = G[G.dim == 'applicant_late_decile_n50']
out['decile'] = dec[['level', 'n', 'wkd', 'L0', 'chg1', 'noise_frac_R1']].round(5).to_dict('records')
nz = dec[dec.L0 > 0]
out['decile_loglog_slope'] = float(np.polyfit(np.log(nz.L0), np.log(nz.chg1.clip(lower=1e-9)), 1)[0])
# C08 E*: 기한 후 7일 초과 정의의 오염
m7 = lag > 7
out['C08_lag_gt7'] = dict(n=int(m7.sum()), R2_not_late=float((m7 & ~L2).sum() / m7.sum()), R2_lag_le7=float((m7 & (l2 <= 7)).sum() / m7.sum()),
                          R1_not_late=float((m7 & ~L1).sum() / m7.sum()))
out['top20_rows_share'] = float((size == 'top20').mean())
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
print(G[G.dim.isin(['size', 'applicant_late_decile_n50'])].round(4).to_string())
json.dump(out, open(os.path.join(O, 'r1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(f'{time.time()-t:.0f}s')

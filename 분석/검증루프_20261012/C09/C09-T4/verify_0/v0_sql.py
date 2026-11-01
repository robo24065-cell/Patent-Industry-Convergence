# C09-T4 verify_0: 보정 지연 피처를 sqlite(mode=ro) 독립 SQL로 다시 계산해 본 계산(s2·s4 산출)과 대조.
#  파이썬 피처 코드(t4lib·n4feat·s2·s4)를 쓰지 않는다. 공휴일 목록만 cache/holidays_kr_2000_2026.csv 를 temp 표로 올린다.
#  기한: 등록일 + (statAnnl−1)년(2/29 등록이면 2/28). 영업일 달력: 토·일·공휴일 아닌 날. R1: 토→+2, 일→+1. R2: 기한 이상 첫 영업일.
#  T4: F22 평가행(L=2022-01-01) 287,578건의 5개 피처 × R1·R2·R3.  N4: final 패널 t0=2023 의 2개 피처 × R1·R2·R3.
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B + r'\수집파이프라인\registry_full.sqlite'
W = B + r'\분석\검증루프_20261012\C09\C09-T4'
T4F = B + r'\분석\모델경쟁_20260927\T4_특허포기예측\R2\runs_final\F22'
N4C = B + r'\분석\모델경쟁_20260927\N4_특허행태_기업폐업조기경보\cache'
t = time.time()
c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
hol = pd.read_csv(os.path.join(W, 'cache', 'holidays_kr_2000_2026.csv'))
c.execute('create temp table hol(d text primary key)')
c.executemany('insert into hol values (?)', [(d,) for d in hol.date])
c.execute('create temp table cal(d text primary key, bus int)')
days = pd.date_range('1999-01-01', '2027-12-31').strftime('%Y-%m-%d')
hs = set(hol.date)
c.executemany('insert into cal values (?, ?)', [(d, int(pd.Timestamp(d).weekday() < 5 and d not in hs)) for d in days])
# 기한 계산 뷰(갱신 납부 seq>0, 특허 10…)
c.execute("""create temp table pv as
 select p.rgst_no, p.seq, cast(coalesce(nullif(p.statAnnl,''), p.lastAnnl) as int) s, p.statAnnl sa,
        substr(p.payDate,1,4)||'-'||substr(p.payDate,5,2)||'-'||substr(p.payDate,7,2) pd,
        printf('%04d', cast(substr(t.rgstDate,1,4) as int) + cast(coalesce(nullif(p.statAnnl,''), p.lastAnnl) as int) - 1) || '-' || substr(t.rgstDate,5,2) || '-' ||
          (case when substr(t.rgstDate,5,4)='0229' then '28' else substr(t.rgstDate,7,2) end) due
 from pay p join patent t on p.rgst_no = t.rgst_no
 where p.seq > 0 and length(p.payDate)=8 and length(t.rgstDate)=8""")
c.execute("""create temp table pv2 as
 select pv.*, cast(strftime('%w', due) as int) w,
   case cast(strftime('%w', due) as int) when 6 then date(due, '+2 days') when 0 then date(due, '+1 day') else due end dueR1,
   (select min(cal.d) from cal where cal.d >= pv.due and cal.bus = 1) dueR2
 from pv""")
c.execute('create index temp.ix_pv2 on pv2(rgst_no)')
print('pv2', c.execute('select count(*) from pv2').fetchone(), f'{time.time() - t:.0f}s', flush=True)
out = {}
# ---------------- T4 F22
ev = pd.read_parquet(os.path.join(T4F, 'ev.parquet'), columns=['pid'])
pt = pd.read_parquet(os.path.join(W, 'cache', 't4_pt.parquet'), columns=['pid', 'rgst_no'])   # pid ↔ 등록번호(트랙 적재 순서)
ev = ev.merge(pt, on='pid', how='left')
c.execute('create temp table evp(pid int, rgst_no text)')
c.executemany('insert into evp values (?, ?)', list(zip(ev.pid.astype(int), ev.rgst_no)))
q = """
 with x as (
   select e.pid, v.seq, v.pd,
     julianday(v.due) - julianday(v.pd) e0, julianday(v.dueR1) - julianday(v.pd) e1, julianday(v.dueR2) - julianday(v.pd) e2
   from evp e join pv2 v on v.rgst_no = e.rgst_no where v.pd <= '2022-01-01'),
 lst as (select pid, max(pd || printf('%06d', seq)) k from x group by pid)
 select x.pid,
   count(*) n_renew,
   sum(e1 < 0) nl_R1, sum(e2 < 0) nl_R2, sum(e0 < -3) nl_R3,
   avg(e1) me_R1, avg(e2) me_R2, avg(e0) me_R3, min(e1) mi_R1, min(e2) mi_R2, min(e0) mi_R3,
   max(case when x.pd || printf('%06d', x.seq) = lst.k then e1 end) la_R1,
   max(case when x.pd || printf('%06d', x.seq) = lst.k then e2 end) la_R2,
   max(case when x.pd || printf('%06d', x.seq) = lst.k then e0 end) la_R3
 from x join lst on x.pid = lst.pid group by x.pid"""
s = pd.read_sql(q, c).set_index('pid')
print('T4 sql', len(s), f'{time.time() - t:.0f}s', flush=True)
xd = os.path.join(W, 'cache', 't4x', 'F22')
res = {}
for r in ['R1', 'R2', 'R3']:
    mine = {f: np.load(os.path.join(xd, f'ev__{f}__{r}.npy')).astype(float) for f in ['n_late', 'late_share', 'mean_early', 'min_early', 'last_early']}
    S = s.reindex(ev.pid.values)
    nr = S.n_renew.fillna(0).values
    sql = {'n_late': S[f'nl_{r}'].fillna(0).values, 'late_share': np.where(nr > 0, S[f'nl_{r}'].values / np.where(nr > 0, nr, 1), np.nan),
           'mean_early': S[f'me_{r}'].values, 'min_early': S[f'mi_{r}'].values, 'last_early': S[f'la_{r}'].values}
    rr = {}
    for f in mine:
        a, b = mine[f], sql[f].astype(float)
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= np.maximum(1e-3, 1e-4 * np.abs(b)))
        rr[f] = dict(match=float(same.mean()), n_mismatch=int((~same).sum()))
    res[r] = rr
out['T4_F22'] = res
print('T4', json.dumps(res, ensure_ascii=False), flush=True)
# ---------------- N4 final t0=2023
P = pd.read_parquet(os.path.join(N4C, 'final_panel.parquet'), columns=['applicantCd', 't0'])
idx = np.where(P.t0.values == 2023)[0]
c.execute('create temp table fa(a text primary key)')
c.executemany('insert or ignore into fa values (?)', [(a,) for a in P.applicantCd.values[idx]])
q2 = """
 with w as (
   select distinct a.applicantCd ac, p.rgst_no, p.seq, p.payDate, cast(p.statAnnl as int) s, t.rgstDate
   from pay p join patent t on p.rgst_no = t.rgst_no join applicant a on a.rgst_no = p.rgst_no join fa on fa.a = a.applicantCd
   where cast(p.payDate as int) > 20211231 and cast(p.payDate as int) <= 20231231 and cast(p.statAnnl as real) >= 4 and length(t.rgstDate)=8),
 v as (
   select ac, substr(payDate,1,4)||'-'||substr(payDate,5,2)||'-'||substr(payDate,7,2) pd,
     printf('%04d', cast(substr(rgstDate,1,4) as int) + s - 1) || '-' || substr(rgstDate,5,2) || '-' || (case when substr(rgstDate,5,4)='0229' then '28' else substr(rgstDate,7,2) end) due
   from w),
 v2 as (select ac, pd, due,
   case cast(strftime('%w', due) as int) when 6 then date(due, '+2 days') when 0 then date(due, '+1 day') else due end dueR1,
   (select min(cal.d) from cal where cal.d >= v.due and cal.bus = 1) dueR2 from v),
 v3 as (select ac, julianday(pd) - julianday(due) d0, julianday(pd) - julianday(dueR1) d1, julianday(pd) - julianday(dueR2) d2 from v2)
 select ac, avg(d1 > 0) ls_R1, avg(d2 > 0) ls_R2, avg(d0 > 3) ls_R3,
   avg(max(d1, 0)) dm_R1, avg(max(d2, 0)) dm_R2, avg(case when d0 > 3 then d0 else 0 end) dm_R3
 from v3 group by ac"""
s2 = pd.read_sql(q2, c).set_index('ac')
print('N4 sql', len(s2), f'{time.time() - t:.0f}s', flush=True)
res2 = {}
for r in ['R1', 'R2', 'R3']:
    mine = pd.read_parquet(os.path.join(W, 'cache', f'n4_late_final_{r}.parquet')).iloc[idx]
    S = s2.reindex(P.applicantCd.values[idx])
    rr = {}
    for f, k in [('late_share_2y', 'ls'), ('delay_mean_2y', 'dm')]:
        a = mine[f].to_numpy(float); b = S[f'{k}_{r}'].to_numpy(float)
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-6 * np.maximum(1, np.abs(b)))
        rr[f] = dict(match=float(same.mean()), n_mismatch=int((~same).sum()))
    res2[r] = rr
out['N4_final_2023'] = res2
print('N4', json.dumps(res2, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(W, 'verify_0', 'v0_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
# ---- 불일치 행 진단
mine = pd.read_parquet(os.path.join(W, 'cache', 'n4_late_final_R2.parquet')).iloc[idx]
S = s2.reindex(P.applicantCd.values[idx])
a = mine['late_share_2y'].to_numpy(float); b = S['ls_R2'].to_numpy(float)
bad = np.where(~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-6)))[0]
diag = []
for i in bad:
    ac = P.applicantCd.values[idx][i]
    rows = pd.read_sql("""select p.rgst_no, p.payDate, p.statAnnl, t.rgstDate from pay p join patent t on p.rgst_no=t.rgst_no join applicant a on a.rgst_no=p.rgst_no
        where a.applicantCd=? and cast(p.payDate as int) > 20211231 and cast(p.payDate as int) <= 20231231 and cast(p.statAnnl as real) >= 4""", c, params=(ac,))
    diag.append(dict(ac_hash=hash(ac) % 10**6, mine=float(a[i]), sql=float(b[i]), n_rows=len(rows), rows=rows.drop(columns=['rgst_no']).to_dict('records')))
print(diag)
out['N4_mismatch_diag'] = diag
json.dump(out, open(os.path.join(W, 'verify_0', 'v0_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)

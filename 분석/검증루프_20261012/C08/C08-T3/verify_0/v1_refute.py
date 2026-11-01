# -*- coding: utf-8 -*-
"""verify_0/v1_refute — 반박 검증(별도 작업자). c08lib·v0_indep 미사용.
레이블을 다른 방식(L 시점 납부완료일 pth_L 기반 + L 이후 연장납부 여부)으로 재구성, 동결 점수 원본 대조,
가중 AUC 기반 군집 부트스트랩(B=1000, seed 2026) 재계산. 출력 verify_0/v1_out.json"""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
FUT = os.path.join(BASE, r"분석\모델경쟁_20260927\T4_특허포기예측\R2\runs_future\FUT")
O = {}
fz = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet'))
pr = pd.read_parquet(os.path.join(FUT, 'pred_20250701.parquet'))
m = fz[['pid', 'p', 'p_base']].merge(pr[['pid', 'p', 'p_base', 'ksic']], on='pid', suffixes=('', '_o'))
O['frozen_vs_orig'] = dict(n_fz=len(fz), n_orig=len(pr), n_join=len(m), max_dp=float((m.p - m.p_o).abs().max()),
                           max_dpb=float((m.p_base - m.p_base_o).abs().max()), y_realized_nonnull=int(pr.y_realized.notna().sum()))
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
P = pd.read_sql("select rowid rid, rgst_no, rgstDate, ksic, cndrtExptnDate from patent order by rowid", con)
P['rd'] = pd.to_datetime(P.rgstDate, format='%Y%m%d', errors='coerce')
P = P[P.rd.notna() & P.rgst_no.str.startswith('10')].reset_index(drop=True)
P['pid'] = np.arange(len(P))
rs = P.iloc[fz.pid.values].reset_index(drop=True)
O['pid_map_ksic_agree'] = float((rs.ksic.astype(str).values == fz.ksic.astype(str).values).mean())
rset = set(rs.rgst_no)
pay = pd.read_sql("select rgst_no, payDate, lastAnnl from pay", con)
pay = pay[pay.rgst_no.isin(rset)].copy()
pay['pd'] = pd.to_datetime(pay.payDate, format='%Y%m%d', errors='coerce'); pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
pay = pay[pay.pd.notna() & pay.l.notna()]
L = pd.Timestamp('2025-07-01'); WE = pd.Timestamp('2025-10-31'); OBS = pd.Timestamp('2026-08-15')
lL = pay[pay.pd <= L].groupby('rgst_no').l.max()
rs['lL'] = rs.rgst_no.map(lL)
def addy(ts, k):
    out = []
    for t, kk in zip(ts, k):
        if pd.isna(kk): out.append(pd.NaT); continue
        d = 28 if (t.month == 2 and t.day == 29) else t.day
        out.append(pd.Timestamp(t.year + int(kk), t.month, d))
    return pd.DatetimeIndex(out)
rs['pthL'] = addy(rs.rd, rs.lL)
due = ((rs.pthL > L) & (rs.pthL <= WE)).values
O['due_win'] = dict(n_mine=int(due.sum()), n_fz=int(fz.due_win.sum()), mismatch=int((due != (fz.due_win.values == 1)).sum()),
                    pthL_le_L=int((rs.pthL <= L).sum()), pthL_nan=int(rs.pthL.isna().sum()))
# 사건: 불납 소멸등록 cd in (L, OBS]
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right", con)
rt = rt[rt.rgst_no.isin(rset)].copy(); rt['cd'] = pd.to_datetime(rt.rgstCsDate, format='%Y%m%d', errors='coerce')
lap = rt[(rt.rgstCsName == '소멸등록') & rt.rgstCsReason.isin(['등록료불납', '연차료불납']) & (rt.cd > L) & (rt.cd <= OBS)]
O['reasons_for_somyeol_after_L'] = rt[(rt.rgstCsName == '소멸등록') & (rt.cd > L)].rgstCsReason.fillna('NA').value_counts().to_dict()
O['names_with_somyeol_like'] = rt[rt.rgstCsName.fillna('').str.contains('소멸') & (rt.cd > L)].rgstCsName.value_counts().to_dict()
first_cd = lap.groupby('rgst_no').cd.min()
rs['cd'] = rs.rgst_no.map(first_cd)
# L 이후 cd 이전 연장납부(lastAnnl > lL)
pa = pay[pay.pd > L].merge(rs[['rgst_no', 'lL', 'cd']], on='rgst_no')
ext = set(pa[(pa.l > pa.lL) & ((pa.pd < pa.cd) | pa.cd.isna())].rgst_no)
rs['ext'] = rs.rgst_no.isin(ext)
rec = rt[rt.rgstCsName.fillna('').str.startswith('회복등록') & (rt.cd <= OBS)].groupby('rgst_no').cd.max()
rs['rec'] = rs.rgst_no.map(rec)
restored = rs.rec.notna() & rs.cd.notna() & (rs.rec >= rs.cd)
yS1 = (due & rs.cd.notna().values & ~rs.ext.values)
y = yS1 & ~restored.values
lb = pd.read_parquet(os.path.join(W, 'cache', 'labels.parquet'))
assert (lb.pid.values == fz.pid.values).all()
O['labels'] = dict(pos_mine=int(y.sum()), pos_s04=int(lb.y.sum()), mismatch=int((y.astype(int) != lb.y.values).sum()),
                   only_mine=int((y & (lb.y.values == 0)).sum()), only_s04=int((~y & (lb.y.values == 1)).sum()),
                   S1_mine=int(yS1.sum()), S1_s04=int(lb.y_S1.sum()), S1_mismatch=int((yS1.astype(int) != lb.y_S1.values).sum()),
                   lapse_cd_in_obs_any=int(rs.cd.notna().sum()), lapse_but_not_due=int((rs.cd.notna().values & ~due).sum()),
                   lapse_due_but_ext=int((rs.cd.notna().values & due & rs.ext.values).sum()))
# 불일치 표본
dif = rs.loc[(y.astype(int) != lb.y.values), ['rgst_no', 'rd', 'lL', 'pthL', 'cd', 'ext', 'rec']].head(10)
O['mismatch_sample'] = dif.astype(str).to_dict('records')
# 창 끝 절단 위험: due 중 소멸 사건 없는 특허의 pthL 분포(10월 말 집중 여부) 및 월별 양성률
mon = rs.pthL.dt.to_period('M').astype(str)
O['pos_rate_by_due_month'] = {k: [int(v.sum()), int(len(v)), round(float(v.mean()), 4)] for k, v in pd.Series(y[due]).groupby(mon[due].values)}
# 소멸등록 시차(cd - pthL) 분포 (양성)
lag = (rs.cd - rs.pthL).dt.days[y]
O['lag_days_pos'] = dict(p50=float(lag.median()), p99=float(lag.quantile(.99)), max=float(lag.max()), min=float(lag.min()))
# ---- 점수
yi = y.astype(int); p, pb, ps = fz.p.values, fz.p_base.values, fz.p_strong.values
A = lambda yy, s, w=None: float(roc_auc_score(yy, s, sample_weight=w)); sk = lambda a, b: (a - b) / (1 - b)
O['scores'] = dict(auc=A(yi, p), auc_base=A(yi, pb), auc_strong=A(yi, ps))
O['scores']['skill'] = sk(O['scores']['auc'], O['scores']['auc_base'])
O['scores']['skill_strong'] = sk(O['scores']['auc'], O['scores']['auc_strong'])
O['scores']['auc_due'] = A(yi[due], p[due]); O['scores']['auc_base_due'] = A(yi[due], pb[due])
O['scores']['skill_due'] = sk(O['scores']['auc_due'], O['scores']['auc_base_due'])
O['scores']['skill_due_strong'] = sk(O['scores']['auc_due'], A(yi[due], ps[due]))
# ---- 군집 부트스트랩: 군집 가중치(다항) → 가중 AUC (동점 반영)
def prep(s):
    u, inv = np.unique(s, return_inverse=True); return inv, len(u)
def wauc(inv, nu, yy, w):
    wp = np.bincount(inv, weights=w * yy, minlength=nu); wn = np.bincount(inv, weights=w * (1 - yy), minlength=nu)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
acd = fz.acd.astype(str).values
cl = np.where(acd == 'NA', 'P' + fz.pid.astype(str).values, acd)
codes, uniq = pd.factorize(cl); K = len(uniq)
ip, nup = prep(p); ib, nub = prep(pb); iS, nuS = prep(ps)
d_idx = np.where(due)[0]
ipd, nupd = prep(p[d_idx]); ibd, nubd = prep(pb[d_idx]); iSd, nuSd = prep(ps[d_idx])
yf = yi.astype(float); yd = yf[d_idx]
assert abs(wauc(ip, nup, yf, np.ones(len(yf))) - O['scores']['auc']) < 1e-9
rng = np.random.default_rng(2026); R = []
for b in range(1000):
    cw = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
    w = cw[codes]; wd = w[d_idx]
    a, ab, aS = wauc(ip, nup, yf, w), wauc(ib, nub, yf, w), wauc(iS, nuS, yf, w)
    ad, abd, aSd = wauc(ipd, nupd, yd, wd), wauc(ibd, nubd, yd, wd), wauc(iSd, nuSd, yd, wd)
    R.append((a, ab, sk(a, ab), sk(a, aS), sk(ad, abd), sk(ad, aSd)))
R = np.array(R)
ci = lambda c: [float(np.percentile(R[:, c], 2.5)), float(np.percentile(R[:, c], 97.5))]
O['boot'] = dict(B=1000, seed=2026, K=int(K), auc_ci=ci(0), auc_base_ci=ci(1), skill_ci=ci(2), skill_strong_ci=ci(3), skill_due_ci=ci(4), skill_due_strong_ci=ci(5))
O['secs'] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(HERE, 'v1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))

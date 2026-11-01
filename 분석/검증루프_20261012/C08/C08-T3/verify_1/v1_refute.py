# -*- coding: utf-8 -*-
"""C08-T3 verify_1 반박 검증 — c08lib/s04/s05/s06 코드 미사용. sqlite mode=ro 직접 SQL.
출력: verify_1/v1_out.json"""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
O = {}


def todt(s):
    return pd.to_datetime(pd.Series(s, dtype=object).astype(str).str.strip(), format='%Y%m%d', errors='coerce').values


def addyears(d, k):
    d = pd.DatetimeIndex(d); k = np.asarray(k, float)
    ok = (~pd.isna(d)) & np.isfinite(k)
    out = np.full(len(d), np.datetime64('NaT'), dtype='datetime64[ns]')
    day = np.where((d.month == 2) & (d.day == 29), 28, d.day)
    if ok.any():
        out[ok] = pd.to_datetime(pd.DataFrame({'year': (d.year[ok] + k[ok]).astype(int), 'month': d.month[ok].astype(int),
                                               'day': day[ok].astype(int)})).values
    return out


P = pd.read_sql("select rowid rid, rgst_no, rgstDate, applDate, cndrtExptnDate from patent order by rowid", con)
P['rd'] = todt(P.rgstDate)
P = P[P.rd.notna() & P.rgst_no.str.startswith('10')].reset_index(drop=True)
P['pid'] = np.arange(len(P))
P['exp'] = todt(P.cndrtExptnDate)
m = P.exp.isna().values
P.loc[m, 'exp'] = addyears(todt(P.loc[m, 'applDate']), np.full(m.sum(), 20))
A = pd.read_sql("select rgst_no, seq, applicantCd, applicantName from applicant", con)
A['seq'] = pd.to_numeric(A.seq, errors='coerce')
A1 = A.sort_values(['rgst_no', 'seq']).drop_duplicates('rgst_no').set_index('rgst_no')
P['acd'] = P.rgst_no.map(A1.applicantCd).fillna('NA'); P['aname'] = P.rgst_no.map(A1.applicantName).fillna('')
idx = pd.Series(P.pid.values, index=P.rgst_no.values)
pay = pd.read_sql("select rgst_no, lastAnnl, payDate from pay", con)
pay['pid'] = pay.rgst_no.map(idx); pay['l'] = pd.to_numeric(pay.lastAnnl, errors='coerce'); pay['pd'] = todt(pay.payDate)
pay = pay[pay.pid.notna() & pay.l.notna() & pay.pd.notna()][['pid', 'l', 'pd']].astype({'pid': 'int64'})
R = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right", con)
R['pid'] = R.rgst_no.map(idx); R = R[R.pid.notna()].copy(); R['pid'] = R.pid.astype('int64'); R['cd'] = todt(R.rgstCsDate)
MAL = ['권리의 말소등록', '말소등록', '직권말소등록']
lap = R[(R.rgstCsName == '소멸등록') & R.rgstCsReason.isin(['등록료불납', '연차료불납']) & R.cd.notna()][['pid', 'cd']].reset_index(drop=True)
mm = lap.reset_index().merge(pay, on='pid'); mm = mm[mm.pd < mm.cd]
lap['lmax'] = lap.index.map(mm.groupby('index').l.max()); lap = lap[lap.lmax.notna()].copy()
lap['E'] = addyears(P.rd.values[lap.pid.values], lap.lmax.values)
rec = R[R.rgstCsName == '회복등록'][['pid', 'cd']].rename(columns={'cd': 'rcd'})


def events(L, W_, cap):
    e = lap[(lap.E > L) & (lap.E <= W_) & (lap.cd <= cap)].copy()
    r = rec[rec.rcd <= cap].groupby('pid').rcd.max()
    e['rcd'] = e.pid.map(r); e['restored'] = e.rcd.notna() & (e.rcd >= e.cd)
    return e


def riskset(L):
    L = pd.Timestamp(L)
    p = pay[pay.pd <= L].groupby('pid').l.max()
    lv = np.zeros(len(P)); lv[p.index.values] = p.values
    pth = np.full(len(P), np.datetime64('NaT'), dtype='datetime64[ns]'); ok = lv > 0
    pth[ok] = addyears(P.rd.values[ok], lv[ok])
    Ln = np.datetime64(L)
    rs = (P.rd.values < Ln) & (pth > Ln) & (P.exp.values > Ln)
    ml = R[R.rgstCsName.isin(MAL) & (R.cd <= L)].pid.unique(); rs[ml] = False
    return np.where(rs)[0]


L = pd.Timestamp('2025-07-01'); Wd = pd.Timestamp('2025-10-31'); CAP = pd.Timestamp('2026-08-15')
fz = pd.read_parquet(os.path.join(W, 'frozen_scores.parquet')); lb = pd.read_parquet(os.path.join(W, 'cache', 'labels.parquet'))
rs = riskset(L)
O['riskset'] = dict(n_indep=len(rs), n_frozen=len(fz), same=bool(len(rs) == len(fz) and (np.sort(rs) == np.sort(fz.pid.values)).all()))
e25 = events(L, Wd, CAP); pos = set(e25[~e25.restored].pid)
y = fz.pid.isin(pos).astype(int).values
O['labels'] = dict(pos=int(y.sum()), mismatch_vs_cache=int((y != lb.y.values).sum()))
acd = P.acd.values[fz.pid.values]
O['acd_mismatch'] = int((acd != fz.acd.values).sum())
vc = pd.Series(acd[acd != 'NA']).value_counts(); top10 = vc.index[:10].tolist()
O['top3'] = [(a, int(vc[a])) for a in top10[:3]]
big10 = np.isin(acd, top10)
O['big10_mismatch_frozen'] = int((big10.astype(int) != fz.big10.values).sum())
p, pb = fz.p.values, fz.p_base.values
auc = lambda yy, s: roc_auc_score(yy, s)
sk = lambda yy, a, b: (auc(yy, a) - auc(yy, b)) / (1 - auc(yy, b))


def sub(mk):
    return dict(n=int(mk.sum()), pos=int(y[mk].sum()), auc=auc(y[mk], p[mk]), auc_base=auc(y[mk], pb[mk]), skill=sk(y[mk], p[mk], pb[mk]))


allm = np.ones(len(y), bool)
O['skill'] = dict(all=sub(allm), excl_big10=sub(~big10), big10=sub(big10))
O['big10_leave_one_out'] = {a: sk(y[big10 & (acd != a)], p[big10 & (acd != a)], pb[big10 & (acd != a)]) for a in top10[:3]}
cl = np.where(acd == 'NA', 'P' + fz.pid.astype(str).values, acd)
codes, uniq = pd.factorize(cl); K = len(uniq)
order = np.argsort(codes, kind='stable'); st = np.searchsorted(codes[order], np.arange(K)); en = np.r_[st[1:], len(codes)]
rng = np.random.default_rng(2026); D1, D2, BX, BI = [], [], [], []
for it in range(400):
    pick = rng.integers(0, K, K)
    ii = np.concatenate([order[st[c]:en[c]] for c in pick])
    yy, pp, bb, gg = y[ii], p[ii], pb[ii], big10[ii]
    try:
        sa = sk(yy, pp, bb); sx = sk(yy[~gg], pp[~gg], bb[~gg]); si = sk(yy[gg], pp[gg], bb[gg])
    except ValueError:
        continue
    D1.append(sx - sa); D2.append(sx - si); BX.append(sx); BI.append(si)
q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
O['boot'] = dict(B=len(D1), seed=2026, excl_minus_all=q(D1), excl_minus_big10=q(D2), excl=q(BX), big10=q(BI),
                 share_excl_gt_big10=float(np.mean(np.array(D2) > 0)))
O['official_big10_skill'] = (0.667 - 0.658) / (1 - 0.658); O['official_rest_skill'] = (0.711 - 0.589) / (1 - 0.589)
pw = 1 - (1 - p) ** (122 / 1096)
due = fz.due_win.values == 1
for nm, a in (('rank1', top10[0]), ('rank2', top10[1]), ('rank3', top10[2])):
    mk = acd == a
    O[nm] = dict(acd=a, name=P.aname.values[fz.pid.values][mk][0], n=int(mk.sum()), events=int(y[mk].sum()), realized=float(y[mk].mean()),
                 mean_p=float(p[mk].mean()), mean_pw=float(pw[mk].mean()), sum_pw=float(pw[mk].sum()), sum_hazwin=float(fz.p_haz_win.values[mk].sum()),
                 due=int((mk & due).sum()), realized_due=float(y[mk & due].mean()) if (mk & due).any() else None,
                 mean_hazwin_due=float(fz.p_haz_win.values[mk & due].mean()) if (mk & due).any() else None,
                 mean_p_due=float(p[mk & due].mean()) if (mk & due).any() else None)
O['all'] = dict(sum_hazwin=float(fz.p_haz_win.sum()), sum_pw=float(pw.sum()), events=int(y.sum()),
                realized_due=float(y[due].mean()), mean_hazwin_due=float(fz.p_haz_win.values[due].mean()))
H = {}
for yy_ in range(2019, 2026):
    Ly = pd.Timestamp(f'{yy_}-07-01'); Wy = pd.Timestamp(f'{yy_}-10-31'); Cy = Ly + pd.Timedelta(days=410)
    r_ = riskset(Ly); ey = events(Ly, Wy, Cy); ey = ey[~ey.restored]
    ac_ = P.acd.values[r_]; ev = np.isin(r_, ey.pid.values)
    H[yy_] = {nm: dict(riskset=int((ac_ == a).sum()), events=int(ev[ac_ == a].sum())) for nm, a in (('rank2', top10[1]), ('rank3', top10[2]))}
    H[yy_]['all'] = dict(riskset=len(r_), events=int(ev.sum()))
O['history'] = H
r3pos = fz.pid.values[(acd == top10[2]) & (y == 1)]
rn = set(P.rgst_no.values[r3pos])
ow = pd.read_sql("select rgst_no, seq, finalOwnerYn, ownerName, ownerRgstCsDate, ownerRgstCsName, ownerRgstCsReason from owner", con)
ow = ow[ow.rgst_no.isin(rn)]
fo = ow[ow.finalOwnerYn == 'Y'].drop_duplicates('rgst_no')
O['rank3_pos_owner'] = dict(n_pos=len(rn), n_with_owner_rows=int(ow.rgst_no.nunique()),
                            final_owner_name_counts=fo.ownerName.value_counts().head(5).to_dict(),
                            reasons=ow.ownerRgstCsReason.value_counts().head(6).to_dict())
e3 = e25[e25.pid.isin(r3pos)]
O['rank3_pos_cd_month'] = pd.DatetimeIndex(e3.cd).strftime('%Y-%m').value_counts().sort_index().to_dict()
O['rank3_pos_E_month'] = pd.DatetimeIndex(e3.E).strftime('%Y-%m').value_counts().sort_index().to_dict()
O['rank3_pos_ksic'] = fz.ksic[fz.pid.isin(r3pos)].astype(str).value_counts().head(5).to_dict()
O['rank3_pos_lmax'] = {str(k): int(v) for k, v in e3.lmax.value_counts().sort_index().items()}
O['secs'] = round(time.time() - t0, 1)
json.dump(O, open(os.path.join(HERE, 'v1_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=1, default=str))

# (특허, 기준일 L) 패널 생성. L = 매년 1월 1일, 2006~2015. 결과창 [L, L+3y) 는 모두 2018-01-01 이전 종료.
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
def P(*a): print(*a, round(time.time() - t0, 1), flush=True)

pat = pd.read_parquet('cache/pat.parquet')
for c in ['applDate', 'rgstDate', 'cndrtExptnDate', 'opnDate', 'pubDate']:
    pat[c] = pd.to_datetime(pat[c], format='%Y%m%d', errors='coerce')
pat['claims'] = pd.to_numeric(pat.claimCount, errors='coerce')

own = pd.read_parquet('cache/own.parquet'); own['d'] = pd.to_datetime(own.d, format='%Y%m%d', errors='coerce')
assert own.d.max() < HOLD_CUT
own['cls'] = own.rs.map(classify); own['nm'] = own.ownerName.map(norm)
own = own.sort_values(['rgst_no', 'd', 'seq'])
g = own.groupby(['rgst_no', 'd'], sort=False).agg(
    prim=('nm', 'first'), raw=('ownerName', 'first'), natl=('ownerNatl', 'first'), n=('nm', 'size'),
    cls=('cls', lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index()
g = g.sort_values(['rgst_no', 'd']).reset_index(drop=True)
g['k'] = g.groupby('rgst_no').cumcount()
g['assignor'] = g.groupby('rgst_no').prim.shift(1)
g.loc[(g.k > 0) & g.cls.isna(), 'cls'] = 'UNK'
g.loc[g.k == 0, 'cls'] = None
ev = g[g.cls.notna()].copy()
P('owner groups', len(g), 'events', len(ev), ev.cls.value_counts().to_dict())

# 출원인 코드 접두 → 이름 유형 사전 (1 국내법인, 2 대학/국공립, 3 연구기관, 4 국내개인, 5 외국법인, 6 외국개인)
app = pd.read_parquet('cache/app.parquet'); app['nm'] = app.applicantName.map(norm); app['pref'] = app.applicantCd.str[:1]
name2pref = app.dropna(subset=['pref']).groupby('nm').pref.agg(lambda s: s.value_counts().index[0]).to_dict()
def otype(nm, raw, natl):
    p = name2pref.get(nm)
    if p is not None and p.isdigit():
        p = int(p)
        if p == 2: return 2 if ('대학' in str(raw) or '산학' in str(raw)) else 7
        if p == 6: return 5
        return p
    return name_type(raw, natl)
g['otype'] = [otype(a, b, c) for a, b, c in zip(g.prim, g.raw, g.natl)]
app0 = app.sort_values(['rgst_no', 'seq']).groupby('rgst_no').agg(
    orig_pref=('pref', 'first'), orig_nm=('nm', 'first'), orig_raw=('applicantName', 'first'), n_app=('nm', 'size')).reset_index()
app0['orig_type'] = pd.to_numeric(app0.orig_pref, errors='coerce').fillna(0).astype(int)
app0.loc[(app0.orig_type == 2) & ~app0.orig_raw.fillna('').str.contains('대학|산학'), 'orig_type'] = 7

rgt = pd.read_parquet('cache/rgt.parquet'); rgt['d'] = pd.to_datetime(rgt.d, format='%Y%m%d', errors='coerce')
assert rgt.d.max() < HOLD_CUT
st = rgt[rgt.nm.isin(['설정등록', '국유특허(실용신안)설정등록', '국유설정등록', '소멸등록', '회복등록', '권리의 말소등록', '말소등록', '직권말소등록'])].copy()
st['dead'] = st.nm.isin(['소멸등록', '권리의 말소등록', '말소등록', '직권말소등록']).astype(int)
st = st.sort_values(['rgst_no', 'd', 'seq'])
lapse = rgt[(rgt.nm == '소멸등록')].copy()
trial = rgt[rgt.nm.str.contains('심판|심결|이의|취소|무효|정정', na=False)]
pay = pd.read_parquet('cache/pay.parquet'); pay['d'] = pd.to_datetime(pay.d, format='%Y%m%d', errors='coerce')
for c in ['lastAnnl', 'statAnnl', 'payAmount']: pay[c] = pd.to_numeric(pay[c], errors='coerce')
lic = pd.concat([pd.read_parquet('cache/lic_e.parquet').assign(ex=1), pd.read_parquet('cache/lic_n.parquet').assign(ex=0)])
lic['d'] = pd.to_datetime(lic.d, format='%Y%m%d', errors='coerce'); lic['e'] = pd.to_datetime(lic.e, format='%Y%m%d', errors='coerce')
inv = pd.read_parquet('cache/inv.parquet'); cof = pd.read_parquet('cache/cof.parquet'); agt = pd.read_parquet('cache/agt.parquet')
has_owner = set(g.rgst_no)
lapse = lapse.dropna(subset=['d']).sort_values('d'); gg = g.sort_values('d')
lapse = pd.merge_asof(lapse, gg[['rgst_no', 'd', 'prim']], on='d', by='rgst_no', direction='backward')

static = pat[['rgst_no', 'ksic', 'applDate', 'rgstDate', 'cndrtExptnDate', 'applTpcd', 'claims', 'ipcCd', 'cpcCd', 'lastDspst', 'rfoexYn', 'title', 'opnDate']]\
    .merge(app0, on='rgst_no', how='left').merge(inv, on='rgst_no', how='left').merge(cof, on='rgst_no', how='left').merge(agt, on='rgst_no', how='left')
static['has_owner'] = static.rgst_no.isin(has_owner).astype(int)
static['inv_is_app'] = [int(isinstance(a, str) and isinstance(b, str) and b != '' and b in a.replace(' ', ''))
                        for a, b in zip(static.inv_names, static.orig_raw.fillna('').str.replace(' ', ''))]
static['ipc4'] = static.ipcCd.fillna('').str[:4]; static['ipc1'] = static.ipcCd.fillna('').str[:1]
static['exam_yrs'] = (static.rgstDate - static.applDate).dt.days / 365.25
static['for_prio'] = static.prio_natl.fillna('').apply(lambda s: int(any(x not in ('대한민국', '') for x in s.split('|'))))
static = static.drop(columns=['inv_names'])
otype_idx = g.set_index(['rgst_no', 'd']).otype

for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1); E = pd.Timestamp(Y + 3, 1, 1); B = pd.Timestamp(Y - 3, 1, 1)
    LABEL_OK = E <= HOLD_CUT
    s = static[(static.rgstDate < L) & ((static.cndrtExptnDate.isna()) | (static.cndrtExptnDate > L))].copy()
    last = st[st.d < L].groupby('rgst_no').dead.last()
    s = s[~s.rgst_no.isin(last[last == 1].index)]
    n_all = len(s); s = s[s.has_owner == 1]; n_noown = n_all - len(s)
    s['L'] = Y
    s['age'] = (L - s.rgstDate).dt.days / 365.25; s['age_appl'] = (L - s.applDate).dt.days / 365.25; s['rgst_year'] = s.rgstDate.dt.year
    gl = g[g.d < L]; cur = gl.groupby('rgst_no').tail(1).set_index('rgst_no')
    s['cur'] = s.rgst_no.map(cur.prim); s['cur_type'] = s.rgst_no.map(cur.otype); s['cur_n'] = s.rgst_no.map(cur.n)
    s['cur'] = s.cur.fillna(s.orig_nm).fillna('')
    s['cur_is_orig'] = (s.cur == s.orig_nm).astype(int)
    el = ev[ev.d < L]
    for c in ['MAIN', 'CORP']:
        cnt = el[el.cls == c].groupby('rgst_no').size(); s[f'prior_{c}'] = s.rgst_no.map(cnt).fillna(0)
    lastch = el.groupby('rgst_no').d.max(); s['yrs_since_change'] = ((L - s.rgst_no.map(lastch)).dt.days / 365.25).fillna(-1)
    port = s.groupby('cur').size(); s['own_port'] = s.cur.map(port)
    e3 = el[(el.d >= B) & (el.cls == 'MAIN')]
    s['own_out3'] = s.cur.map(e3.groupby('assignor').rgst_no.nunique()).fillna(0)
    s['own_out3_days'] = s.cur.map(e3.groupby('assignor').d.nunique()).fillna(0)
    s['own_in3'] = s.cur.map(e3.groupby('prim').rgst_no.nunique()).fillna(0)
    s['own_out_all'] = s.cur.map(el[el.cls == 'MAIN'].groupby('assignor').rgst_no.nunique()).fillna(0)
    s['own_corp3'] = s.cur.map(el[(el.d >= B) & (el.cls == 'CORP')].groupby('assignor').rgst_no.nunique()).fillna(0)
    l3 = lapse[(lapse.d < L) & (lapse.d >= B)]
    s['own_lapse3'] = s.cur.map(l3.groupby('prim').size()).fillna(0)
    l1 = lapse[(lapse.d < L) & (lapse.d >= pd.Timestamp(Y - 1, 1, 1))]
    s['own_lapse1'] = s.cur.map(l1.groupby('prim').size()).fillna(0)
    s['own_lapse_rate'] = s.own_lapse3 / (s.own_port + s.own_lapse3)
    s['own_out_rate'] = s.own_out3 / (s.own_port + s.own_out3)
    kev = e3.merge(pat[['rgst_no', 'ksic']], on='rgst_no').groupby('ksic').rgst_no.nunique()
    s['ksic_rate3'] = s.ksic.map(kev).fillna(0) / s.ksic.map(s.groupby('ksic').size())
    pl = pay[pay.d < L].sort_values(['rgst_no', 'd'])
    pg = pl.groupby('rgst_no').agg(n_pay=('d', 'size'), paid_thru=('lastAnnl', 'max'), last_pay=('d', 'max'))
    s = s.merge(pg, left_on='rgst_no', right_index=True, how='left')
    s['prepaid'] = s.paid_thru - s.age
    s['days_since_pay'] = (L - s.last_pay).dt.days
    pl2 = pl.merge(pg[['last_pay']], left_on='rgst_no', right_index=True); pl2 = pl2[pl2.d == pl2.last_pay]
    sp = pl2.groupby('rgst_no').agg(la=('lastAnnl', 'max'), sa=('statAnnl', 'min'), amt=('payAmount', 'sum'))
    sp['amt_per_yr'] = sp.amt / (sp.la - sp.sa + 1).clip(lower=1); sp['multi'] = (sp.la - sp.sa)
    s = s.merge(sp[['amt_per_yr', 'multi']], left_on='rgst_no', right_index=True, how='left')
    lc = lic[lic.d < L]; s['lic_any'] = s.rgst_no.isin(lc.rgst_no).astype(int)
    s['lic_active'] = s.rgst_no.isin(lc[(lc.e.isna()) | (lc.e > L)].rgst_no).astype(int)
    s['lic_excl'] = s.rgst_no.isin(lc[lc.ex == 1].rgst_no).astype(int)
    tr = trial[trial.d < L].groupby('rgst_no').size(); s['n_trial'] = s.rgst_no.map(tr).fillna(0)
    # 레이블 (L <= d < L+3y)
    ew = ev[(ev.d >= L) & (ev.d < E)]
    for c in ['MAIN', 'CORP', 'INH', 'OTHER']:
        s[f'y_{c}'] = s.rgst_no.isin(ew[ew.cls == c].rgst_no).astype(np.int8)
    s['y_ANY'] = s.rgst_no.isin(ew.rgst_no).astype(np.int8)
    fm = ew[ew.cls == 'MAIN'].sort_values('d').groupby('rgst_no').first()
    fm['atype'] = otype_idx.reindex(list(zip(fm.index, fm.d))).values
    s['y_assignee_type'] = s.rgst_no.map(fm.atype); s['y_date'] = s.rgst_no.map(fm.d); s['y_assignee'] = s.rgst_no.map(fm.prim)
    if not LABEL_OK:
        for c in ['y_MAIN', 'y_CORP', 'y_INH', 'y_OTHER', 'y_ANY']: s[c] = np.int8(-1)
        s['y_assignee_type'] = np.nan; s['y_date'] = pd.NaT; s['y_assignee'] = None
    s = s.drop(columns=['last_pay', 'applDate', 'cndrtExptnDate', 'opnDate', 'orig_pref'])
    s.to_parquet(f'cache/panel_L{Y}.parquet', index=False)
    P(Y, 'alive', n_all, 'no_owner_excluded', n_noown, 'n', len(s), 'y_MAIN', int(s.y_MAIN.sum()), round(s.y_MAIN.mean(), 4),
      'CORP', int(s.y_CORP.sum()), 'INH', int(s.y_INH.sum()), 'OTHER', int(s.y_OTHER.sum()), 'ANY', int(s.y_ANY.sum()))

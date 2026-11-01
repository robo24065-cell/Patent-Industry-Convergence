# 원점별 위험집합 + M0(비산업) 피처 + 창 안 첫 인정 이전 사건(양수인 정보). 산업 배정(KSIC)과 무관한 부분만.
# 원점: 2012–2024(2012–2014·2023–2024는 피처 (iv)용 사건 계산만), 개발 2015–2022, 잠금 2025(위험집합·피처만, 레이블 없음).
import sys, time, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
t = time.time(); R = {}
def P(*a): print(*a, round(time.time() - t, 1), flush=True)
pat = pd.read_parquet(C + r'\pat.parquet')
pat['claims'] = pd.to_numeric(pat.claimCount, errors='coerce')
pat['n_ipc'] = pat.ipcCd.fillna('').map(lambda s: len([x for x in s.split('|') if x.strip()]))
for c in ['applDate', 'rgstDate', 'cndrtExptnDate']:
    pat[c] = pd.to_datetime(pat[c], format='%Y%m%d', errors='coerce')
own = pd.read_parquet(C + r'\own.parquet')
assert own.d.max() < EVENT_CUT
own['dd'] = pd.to_datetime(own.d, format='%Y%m%d')
own['nn'] = own.ownerName.map(norm)
own['acc'] = (own.nm == '양도') & (own.rs == '권리의 전부이전등록')
own['evt'] = own.nm.notna()
own = own.sort_values(['rgst_no', 'dd', 'seq'])
g = own.groupby(['rgst_no', 'dd'], sort=False).agg(prim=('nn', 'first'), raw=('ownerName', 'first'), natl=('ownerNatl', 'first'),
                                                    n=('nn', 'size'), acc=('acc', 'max'), evt=('evt', 'max')).reset_index()
g = g.sort_values(['rgst_no', 'dd']).reset_index(drop=True)
g['k'] = g.groupby('rgst_no').cumcount()
g['assignor'] = g.groupby('rgst_no').prim.shift(1); g['assignor_raw'] = g.groupby('rgst_no').raw.shift(1)
app = pd.read_parquet(C + r'\app.parquet'); app['nn'] = app.applicantName.map(norm); app['pref'] = app.applicantCd.str[:1]
name2pref = app.dropna(subset=['pref']).groupby('nn').pref.agg(lambda s: s.value_counts().index[0]).to_dict()
def otype(nm, raw, natl):
    p = name2pref.get(nm)
    if p is not None and p.isdigit():
        p = int(p)
        if p == 2: return 2 if ('대학' in str(raw) or '산학' in str(raw)) else 7
        return p          # 1 국내법인 3 연구기관 4 국내개인 5 외국법인 6 외국개인
    return name_type(raw, natl)
g['otype'] = [otype(a, b, c) for a, b, c in zip(g.prim, g.raw, g.natl)]
P('groups', len(g))
# 인정 사건: k>=1, 전부 양도, 같은 실체·같은 그룹 제외
def same_entity(a, b):
    if not isinstance(a, str) or not isinstance(b, str) or a == '' or b == '':
        return False
    if a == b: return True
    return (len(a) >= 3 and len(b) >= 3) and (a in b or b in a)
ev = g[(g.k > 0) & g.acc].copy()
ev['same_ent'] = [same_entity(a, b) for a, b in zip(ev.prim, ev.assignor)]
ga = ev.raw.map(group_of); gb = ev.assignor_raw.map(group_of)
ev['same_grp'] = ga.notna() & (ga == gb)
R['acc_groups_all'] = len(ev); R['excl_same_entity'] = int(ev.same_ent.sum()); R['excl_same_group'] = int((ev.same_grp & ~ev.same_ent).sum())
ev = ev[~ev.same_ent & ~ev.same_grp].copy()
ev['bundle'] = ev.groupby(['prim', 'dd']).rgst_no.transform('size')
ev = ev.merge(pat[['rgst_no', 'claims']], on='rgst_no', how='left')
ev[['rgst_no', 'dd', 'prim', 'raw', 'otype', 'assignor', 'bundle']].to_parquet(C + r'\events_acc.parquet', index=False)
P('acc events', len(ev))
# 상태(생존)
rgt = pd.read_parquet(C + r'\rgt.parquet'); assert rgt.d.max() < EVENT_CUT
rgt['dd'] = pd.to_datetime(rgt.d, format='%Y%m%d', errors='coerce')
st = rgt[rgt.nm.isin(STATUS)].copy(); st['dead'] = st.nm.isin(DEAD).astype(int); st = st.sort_values(['rgst_no', 'dd', 'seq'])
lapse = rgt[rgt.nm == '소멸등록'].dropna(subset=['dd']).sort_values('dd')
gg = g.sort_values('dd')
lapse = pd.merge_asof(lapse, gg[['rgst_no', 'dd', 'prim']], on='dd', by='rgst_no', direction='backward')
trial = rgt[rgt.nm.str.contains('심판|심결|이의|취소|무효|정정', na=False)]
pay = pd.read_parquet(C + r'\pay.parquet'); pay['dd'] = pd.to_datetime(pay.d, format='%Y%m%d', errors='coerce'); pay['lastAnnl'] = pd.to_numeric(pay.lastAnnl, errors='coerce')
lic = pd.read_parquet(C + r'\lic.parquet'); lic['dd'] = pd.to_datetime(lic.d, format='%Y%m%d', errors='coerce')
inv = pd.read_parquet(C + r'\inv.parquet'); cof = pd.read_parquet(C + r'\cof.parquet'); agt = pd.read_parquet(C + r'\agt.parquet')
a0 = app.sort_values(['rgst_no', 'seq']).groupby('rgst_no').agg(orig_pref=('pref', 'first'), orig_nm=('nn', 'first'), orig_raw=('applicantName', 'first')).reset_index()
a0['orig_type'] = pd.to_numeric(a0.orig_pref, errors='coerce').fillna(0).astype(int)
static = pat[['rgst_no', 'applNo', 'ksic', 'applDate', 'rgstDate', 'cndrtExptnDate', 'claims', 'n_ipc']].merge(a0[['rgst_no', 'orig_nm', 'orig_type']], on='rgst_no', how='left')\
    .merge(inv, on='rgst_no', how='left').merge(cof, on='rgst_no', how='left').merge(agt, on='rgst_no', how='left')
has_owner = set(g.rgst_no)
for Y in range(2012, LOCK + 1):
    L = pd.Timestamp(Y, 1, 1); E = pd.Timestamp(Y + 1, 1, 1); B3 = pd.Timestamp(Y - 3, 1, 1)
    s = static[(static.rgstDate < L) & (static.cndrtExptnDate.isna() | (static.cndrtExptnDate > L))].copy()
    last = st[st.dd < L].groupby('rgst_no').dead.last(); s = s[~s.rgst_no.isin(last[last == 1].index)]
    n_alive = len(s); s = s[s.rgst_no.isin(has_owner)]
    s['t0'] = Y
    s['age'] = (L - s.rgstDate).dt.days / 365.25; s['age_appl'] = (L - s.applDate).dt.days / 365.25; s['rg_year'] = s.rgstDate.dt.year
    s['log_claims'] = np.log1p(s.claims)
    gl = g[g.dd < L]; cur = gl.groupby('rgst_no').tail(1).set_index('rgst_no')
    s['cur'] = s.rgst_no.map(cur.prim).fillna(''); s['cur_type'] = s.rgst_no.map(cur.otype); s['cur_n'] = s.rgst_no.map(cur.n)
    s['cur_is_orig'] = (s.cur == s.orig_nm).astype(int)
    el = ev[ev.dd < L]
    s['prior_trf'] = s.rgst_no.map(el.groupby('rgst_no').size()).fillna(0)
    lastch = gl.groupby('rgst_no').dd.max(); s['yrs_since_change'] = (L - s.rgst_no.map(lastch)).dt.days / 365.25
    port = s.groupby('cur').size(); s['own_port'] = s.cur.map(port); s['log_own_port'] = np.log1p(s.own_port)
    e3 = el[el.dd >= B3]
    s['own_out3_rate'] = s.cur.map(e3.groupby('assignor').rgst_no.nunique()).fillna(0) / s.own_port
    s['own_in3_rate'] = s.cur.map(e3.groupby('prim').rgst_no.nunique()).fillna(0) / s.own_port
    l3 = lapse[(lapse.dd < L) & (lapse.dd >= B3)]
    s['own_lapse3_rate'] = s.cur.map(l3.groupby('prim').size()).fillna(0) / s.own_port
    pl = pay[pay.dd < L]; pg = pl.groupby('rgst_no').agg(n_pay=('dd', 'size'), paid_thru=('lastAnnl', 'max'), last_pay=('dd', 'max'))
    s = s.merge(pg, left_on='rgst_no', right_index=True, how='left')
    s['yrs_since_pay'] = (L - s.last_pay).dt.days / 365.25
    lc = lic[lic.dd < L]
    s['n_lic_excl'] = s.rgst_no.map(lc[lc.ex == 1].groupby('rgst_no').size()).fillna(0)
    s['n_lic_non'] = s.rgst_no.map(lc[lc.ex == 0].groupby('rgst_no').size()).fillna(0)
    s['n_trial'] = s.rgst_no.map(trial[trial.dd < L].groupby('rgst_no').size()).fillna(0)
    # 창 안 첫 인정 사건(잠금 원점은 계산하지 않음)
    if Y < LOCK:
        assert E <= pd.Timestamp(EVENT_CUT)
        ew = ev[(ev.dd >= L) & (ev.dd < E)].sort_values('dd').groupby('rgst_no').first()
        for c, nc in [('dd', 'ev_date'), ('prim', 'buyer'), ('raw', 'buyer_raw'), ('otype', 'buyer_type'), ('assignor', 'seller'), ('bundle', 'bundle')]:
            s[nc] = s.rgst_no.map(ew[c])
        s['has_ev'] = s.ev_date.notna().astype(np.int8)
    s = s.drop(columns=['applDate', 'cndrtExptnDate', 'rgstDate', 'last_pay', 'claims', 'own_port'])
    s.to_parquet(C + rf'\base_{Y}.parquet', index=False)
    R[f'n_{Y}'] = {'alive': n_alive, 'risk': len(s)}
    P(Y, n_alive, len(s))
R['sec'] = round(time.time() - t, 1)
json.dump(R, open(C + r'\s3a_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

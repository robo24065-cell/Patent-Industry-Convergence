# 추가 피처(권리자 단기 동학·출원 모멘텀·포트폴리오 핵심/비핵심). 모두 L 이전 정보만. → cache/extra_L{Y}.parquet
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
def P(*a): print(*a, round(time.time() - t0, 1), flush=True)

pat = pd.read_parquet('cache/pat.parquet', columns=['rgst_no', 'ksic', 'applDate', 'rgstDate', 'opnDate', 'pubDate', 'ipcCd'])
for c in ['applDate', 'rgstDate', 'opnDate', 'pubDate']:
    pat[c] = pd.to_datetime(pat[c], format='%Y%m%d', errors='coerce')
# 공개 시점 = 공개일·공고일·등록일 중 가장 이른 날 (이 날 이후에야 존재를 알 수 있음)
pat['known'] = pat[['opnDate', 'pubDate', 'rgstDate']].min(axis=1)
pat['ipc4'] = pat.ipcCd.fillna('').str[:4]
app = pd.read_parquet('cache/app.parquet', columns=['rgst_no', 'seq', 'applicantName'])
app['nm'] = app.applicantName.map(norm)
pa = app.merge(pat[['rgst_no', 'applDate', 'known']], on='rgst_no')
first_seen = pa.groupby('nm').applDate.min()

own = pd.read_parquet('cache/own.parquet'); own['d'] = pd.to_datetime(own.d, format='%Y%m%d', errors='coerce')
assert own.d.max() < HOLD_CUT
own['cls'] = own.rs.map(classify); own['nm'] = own.ownerName.map(norm)
own = own.sort_values(['rgst_no', 'd', 'seq'])
g = own.groupby(['rgst_no', 'd'], sort=False).agg(prim=('nm', 'first'), cls=('cls', lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index()
g = g.sort_values(['rgst_no', 'd']); g['assignor'] = g.groupby('rgst_no').prim.shift(1)
g['k'] = g.groupby('rgst_no').cumcount(); g.loc[g.k == 0, 'cls'] = None
ev = g[g.cls == 'MAIN'].merge(pat[['rgst_no', 'ksic', 'ipc4']], on='rgst_no')
rgt = pd.read_parquet('cache/rgt.parquet'); rgt['d'] = pd.to_datetime(rgt.d, format='%Y%m%d', errors='coerce')
lapse = rgt[rgt.nm == '소멸등록'].dropna(subset=['d']).sort_values('d')
lapse = pd.merge_asof(lapse, g.sort_values('d')[['rgst_no', 'd', 'prim']], on='d', by='rgst_no', direction='backward')

for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1)
    s = pd.read_parquet(f'cache/panel_L{Y}.parquet', columns=['rgst_no', 'cur', 'ksic', 'ipc4', 'own_lapse3', 'own_lapse1'])
    el = ev[ev.d < L]
    for nm, lo in [('own_out1', pd.Timestamp(Y - 1, 1, 1)), ('own_out6m', L - pd.Timedelta(days=182))]:
        s[nm] = s.cur.map(el[el.d >= lo].groupby('assignor').rgst_no.nunique()).fillna(0)
    s['own_in1'] = s.cur.map(el[el.d >= pd.Timestamp(Y - 1, 1, 1)].groupby('prim').rgst_no.nunique()).fillna(0)
    s['own_lapse_trend'] = (s.own_lapse1 + 1) / (s.own_lapse3 / 3 + 1)
    # 출원 모멘텀 (원출원인 이름 기준, L 이전에 공개된 출원만)
    k = pa[pa.known < L]
    a3 = k[k.applDate >= pd.Timestamp(Y - 3, 1, 1)].groupby('nm').rgst_no.nunique()
    a6 = k[(k.applDate >= pd.Timestamp(Y - 6, 1, 1)) & (k.applDate < pd.Timestamp(Y - 3, 1, 1))].groupby('nm').rgst_no.nunique()
    s['own_app3'] = s.cur.map(a3).fillna(0); s['own_app_prev'] = s.cur.map(a6).fillna(0)
    s['own_app_mom'] = np.log((s.own_app3 + 1) / (s.own_app_prev + 1))
    s['own_years'] = ((L - s.cur.map(first_seen)).dt.days / 365.25).fillna(-1)
    # 포트폴리오 핵심성: 같은 권리자 보유 살아있는 특허 중 같은 KSIC·IPC4 비율
    port = s.groupby('cur').size()
    s['own_ksic_share'] = s.groupby(['cur', 'ksic']).rgst_no.transform('size') / s.cur.map(port)
    s['own_ipc_share'] = s.groupby(['cur', 'ipc4']).rgst_no.transform('size') / s.cur.map(port)
    s['own_n_ksic'] = s.cur.map(s.groupby('cur').ksic.nunique())
    # IPC4 최근 거래율 (L-3 ~ L)
    e3 = el[el.d >= pd.Timestamp(Y - 3, 1, 1)]
    s['ipc_rate3'] = s.ipc4.map(e3.groupby('ipc4').rgst_no.nunique()).fillna(0) / s.ipc4.map(s.groupby('ipc4').size())
    s['ksic_rate1'] = s.ksic.map(el[el.d >= pd.Timestamp(Y - 1, 1, 1)].groupby('ksic').rgst_no.nunique()).fillna(0) / s.ksic.map(s.groupby('ksic').size())
    s.drop(columns=['cur', 'ksic', 'ipc4', 'own_lapse3', 'own_lapse1']).assign(L=Y).to_parquet(f'cache/extra_L{Y}.parquet', index=False)
    P(Y, len(s))

# [v30·v32 피처] 모두 L 이전 사건만 → cache/mkt_L{Y}.parquet
#  (v30) 권리자 매각 성향(수축 추정): 최근 5년 양도 유출 특허 수를 보유+유출로 나눈 비율을 전체 평균쪽으로 수축(가상표본 a=20),
#        마지막 매각 후 경과연수, 과거 매각 경험 여부  — 원시 건수(v16)가 아니라 규모 보정·수축된 성향
#  (v32) 기술시장 깊이(유동성): 같은 KSIC·IPC4 에서 최근 3년 양도의 서로 다른 양수인 수·양도인 수(생존 특허 1천 건당),
#        양수인 중 개인·외국·IP관리법인 비중
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
pat = pd.read_parquet('cache/pat.parquet', columns=['rgst_no', 'ksic', 'ipcCd']); pat['ipc4'] = pat.ipcCd.fillna('').str[:4]
own = pd.read_parquet('cache/own.parquet'); own['d'] = pd.to_datetime(own.d, format='%Y%m%d', errors='coerce')
assert own.d.max() < HOLD_CUT
own['cls'] = own.rs.map(classify); own['nm'] = own.ownerName.map(norm)
own = own.sort_values(['rgst_no', 'd', 'seq'])
g = own.groupby(['rgst_no', 'd'], sort=False).agg(prim=('nm', 'first'), raw=('ownerName', 'first'), natl=('ownerNatl', 'first'),
                                                   cls=('cls', lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index()
g = g.sort_values(['rgst_no', 'd']); g['assignor'] = g.groupby('rgst_no').prim.shift(1)
g['k'] = g.groupby('rgst_no').cumcount(); g.loc[g.k == 0, 'cls'] = None
ev = g[g.cls == 'MAIN'].merge(pat[['rgst_no', 'ksic', 'ipc4']], on='rgst_no')
oa = pd.read_parquet('cache/owner_attr2.parquet').set_index('nm')
ev['b_for'] = ev.prim.map(oa.natl_grp).fillna(0).isin([2, 3, 4, 5, 6, 7]).astype(int)
ev['b_ip'] = ev.prim.map(oa.ip_entity).fillna(0).astype(int)
ev['b_ind'] = [int(name_type(r, n) == 4) for r, n in zip(ev.raw, ev.natl)]
for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1)
    s = pd.read_parquet(f'cache/panel_L{Y}.parquet', columns=['rgst_no', 'cur', 'ksic', 'ipc4', 'own_port'])
    el = ev[ev.d < L]; e5 = el[el.d >= pd.Timestamp(Y - 5, 1, 1)]; e3 = el[el.d >= pd.Timestamp(Y - 3, 1, 1)]
    out5 = s.cur.map(e5.groupby('assignor').rgst_no.nunique()).fillna(0)
    prior = e5.rgst_no.nunique() / max(1, len(s) + e5.rgst_no.nunique())
    a = 20.0
    s['sale_prop5'] = (out5 + a * prior) / (out5 + s.own_port + a)
    s['sale_ever'] = s.cur.isin(el.assignor).astype(int)
    s['yrs_since_sale'] = ((L - s.cur.map(el.groupby('assignor').d.max())).dt.days / 365.25).fillna(99)
    for key in ['ksic', 'ipc4']:
        alive = s.groupby(key).size()
        s[f'{key}_buyers3'] = s[key].map(e3.groupby(key).prim.nunique()).fillna(0) / s[key].map(alive) * 1000
        s[f'{key}_sellers3'] = s[key].map(e3.groupby(key).assignor.nunique()).fillna(0) / s[key].map(alive) * 1000
    for c in ['b_for', 'b_ip', 'b_ind']:
        s[f'ksic_{c}3'] = s.ksic.map(e3.groupby('ksic')[c].mean())
    s.drop(columns=['cur', 'ksic', 'ipc4', 'own_port']).assign(L=Y).to_parquet(f'cache/mkt_L{Y}.parquet', index=False)
    print(Y, len(s), 'sale_ever', round(s.sale_ever.mean(), 3), round(time.time() - t0, 1), flush=True)

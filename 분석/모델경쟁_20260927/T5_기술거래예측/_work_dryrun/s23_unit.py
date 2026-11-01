# 사업부 단위 매각 신호: 같은 권리자가 L 이전 3년/1년 안에 '같은 IPC4·같은 KSIC' 특허를 판 건수, 같은 기술의 불납 건수.
# 상대 피처: 권리자 포트폴리오 내 이 특허의 상대 연령·상대 청구항. → cache/unit_L{Y}.parquet
import sys, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
t0 = time.time()
pat = pd.read_parquet('cache/pat.parquet', columns=['rgst_no', 'ksic', 'ipcCd'])
pat['ipc4'] = pat.ipcCd.fillna('').str[:4]; pat['ipc7'] = pat.ipcCd.fillna('').str.split('/').str[0].str.replace(' ', '')
own = pd.read_parquet('cache/own.parquet'); own['d'] = pd.to_datetime(own.d, format='%Y%m%d', errors='coerce')
assert own.d.max() < HOLD_CUT
own['cls'] = own.rs.map(classify); own['nm'] = own.ownerName.map(norm)
own = own.sort_values(['rgst_no', 'd', 'seq'])
g = own.groupby(['rgst_no', 'd'], sort=False).agg(prim=('nm', 'first'), cls=('cls', lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index()
g = g.sort_values(['rgst_no', 'd']); g['assignor'] = g.groupby('rgst_no').prim.shift(1)
g['k'] = g.groupby('rgst_no').cumcount(); g.loc[g.k == 0, 'cls'] = None
ev = g[g.cls == 'MAIN'].merge(pat, on='rgst_no')
rgt = pd.read_parquet('cache/rgt.parquet'); rgt['d'] = pd.to_datetime(rgt.d, format='%Y%m%d', errors='coerce')
lapse = rgt[rgt.nm == '소멸등록'].dropna(subset=['d']).sort_values('d')
lapse = pd.merge_asof(lapse, g.sort_values('d')[['rgst_no', 'd', 'prim']], on='d', by='rgst_no', direction='backward').merge(pat, on='rgst_no')
for Y in range(2003, LMAX + 1):
    L = pd.Timestamp(Y, 1, 1); B3 = pd.Timestamp(Y - 3, 1, 1); B1 = pd.Timestamp(Y - 1, 1, 1)
    s = pd.read_parquet(f'cache/panel_L{Y}.parquet', columns=['rgst_no', 'cur', 'ksic', 'age', 'claims', 'own_port'])
    s = s.merge(pat[['rgst_no', 'ipc4', 'ipc7']], on='rgst_no', how='left')
    e3 = ev[(ev.d < L) & (ev.d >= B3)]; e1 = e3[e3.d >= B1]
    for nm, e, key in [('u_ipc_out3', e3, 'ipc4'), ('u_ipc_out1', e1, 'ipc4'), ('u_ksic_out3', e3, 'ksic'), ('u_grp_out3', e3, 'ipc7')]:
        c = e.groupby(['assignor', key]).rgst_no.nunique()
        c.index.names = ['cur', key]
        s[nm] = s.set_index(['cur', key]).index.map(c).fillna(0).values if len(c) else 0.0
    l3 = lapse[(lapse.d < L) & (lapse.d >= B3)]
    c = l3.groupby(['prim', 'ipc4']).size(); c.index.names = ['cur', 'ipc4']
    s['u_ipc_lapse3'] = s.set_index(['cur', 'ipc4']).index.map(c).fillna(0).values
    # 같은 권리자·같은 IPC4 살아있는 특허 수 대비
    s['u_ipc_alive'] = s.groupby(['cur', 'ipc4']).rgst_no.transform('size')
    s['u_ipc_out_share'] = s.u_ipc_out3 / (s.u_ipc_out3 + s.u_ipc_alive)
    grp = s.groupby('cur')
    s['rel_age'] = s.age - grp.age.transform('mean')
    s['rel_claims'] = s.claims - grp.claims.transform('mean')
    s[['rgst_no', 'u_ipc_out3', 'u_ipc_out1', 'u_ksic_out3', 'u_grp_out3', 'u_ipc_lapse3', 'u_ipc_alive', 'u_ipc_out_share', 'rel_age', 'rel_claims']]\
        .assign(L=Y).to_parquet(f'cache/unit_L{Y}.parquet', index=False)
    print(Y, len(s), round(time.time() - t0, 1), flush=True)

# s2: 제공 KSIC 정의로 개발 원점 t0=2012–2022 P1 집계 + P3 행 표 + 편향 입력. 2025 이후 owner 행은 s1에서 이미 제외.
import sys, os, json, time
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib_cd import *
T0 = time.time()
own, pat, app, docs, orig = load()
assert own.d.max() < '20250101'
S = seller_info(own)
own = own.join(S)
print('seller', round(time.time() - T0), flush=True)
doc_lab = docs.ksic.where(docs.ksic.isin(KSIC50))
pat_lab = dict(zip(pat.rgst_no, pat.ksic))
link = pd.read_parquet(MC + r'\_shared\ext\firm_nps_applicant_link.parquet', columns=['applicantCd', 'nps_hit']).drop_duplicates('applicantCd')
L = set(link[link.nps_hit.astype(bool)].applicantCd)
panel = pd.read_parquet(MC + r'\_shared\ext\firm_nps_firm_panel.parquet', columns=['applicantCd', 'snap', 'n_emp'])
stats, P, rows, cells = [], [], [], []
for t0 in DEV:
    st, pos, blocked, F, w = build(t0, own, docs, app, orig, doc_lab, pat_lab, keep_rows=True)
    T = f'{t0}1231'
    # 매입자 규모: t0 말 문서 수
    dc = docs[(docs.vis <= T) & docs.cd.isin(F)].groupby('cd').appl_no.nunique()
    pos['buyer_docs'] = pos.cd.map(dc).values
    pos['nps'] = pos.cd.isin(L)
    if t0 >= 2015:
        pe = panel[panel.snap == f'{t0}-12'].drop_duplicates('applicantCd').set_index('applicantCd').n_emp
        pos['n_emp'] = pos.cd.map(pe).values
    else:
        pos['n_emp'] = np.nan
    pos['group'] = own.loc[pos.ix, 'group'].values
    pos['seller_type'] = own.loc[pos.ix, 'seller_type'].values
    pos['rg_in_window'] = (own.loc[pos.ix, 'rgstDate'].values > T)
    P.append(pos)
    # 칸(KSIC×t0) 연결·비연결 단위 수
    Fl = F & L
    bl = pd.DataFrame(list(blocked), columns=['cd', 'k'])
    bl = bl[bl.cd.isin(F)]
    bl['lk'] = bl.cd.isin(L)
    bk = bl.groupby('k').agg(nb=('cd', 'size'), nbl=('lk', 'sum'))
    pk = pos.groupby('k').agg(npos=('cd', 'size'), nposl=('nps', 'sum'))
    for k in KSIC50:
        nb = bk.nb.get(k, 0); nbl = bk.nbl.get(k, 0)
        cells.append(dict(t0=t0, k=k, units=len(F) - nb, units_l=len(Fl) - nbl, pos=int(pk.npos.get(k, 0)), pos_l=int(pk.nposl.get(k, 0))))
    st['eligible_firms_nps'] = len(Fl)
    st['wall'] = round(time.time() - T0)
    stats.append(st)
    # P3 행: 각 행을 한 번만(창 첫 12개월; t0=2022는 2024-01~07도)
    yr = w.d.str[:4].astype(int)
    keep = (yr == t0 + 1) | ((t0 == 2022) & (yr == t0 + 2))
    ww = w[keep].copy(); ww['t0'] = t0
    ww['group'] = own.loc[ww.index, 'group'].values; ww['seller_type'] = own.loc[ww.index, 'seller_type'].values
    rows.append(ww[['t0', 'rgst_no', 'd', 'cs', 'corp', 'inst', 'bulk', 'k', 'map', 'cd', 'natl', 'group', 'seller_type']])
    print(st, flush=True)
P = pd.concat(P, ignore_index=True); R = pd.concat(rows, ignore_index=True); CE = pd.DataFrame(cells)
P.drop(columns=['ix']).to_parquet(C + r'\dev_pos.parquet', index=False)
R.to_parquet(C + r'\dev_rows.parquet', index=False)
CE.to_parquet(C + r'\dev_cells.parquet', index=False)
json.dump(stats, open(C + r'\dev_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=int)
print('done', round(time.time() - T0))

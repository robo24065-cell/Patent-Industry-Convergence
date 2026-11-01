# -*- coding: utf-8 -*-
"""의미 검정 1) 한국 HS6 수출(R2global, 재사용) 2) ISO 표준 발행 — 특허 성장과의 선행 관계 (≤2016)."""
import sys; sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext')
import pandas as pd, numpy as np, json
from trade_lib import *
BASE = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
mp = pd.read_csv(BASE + r'\분석\fusion\ksic_hs6_map.csv', dtype=str, encoding='utf-8-sig'); mp['hs6'] = mp.hs6.str.zfill(6)
res = []
# --- 1) 한국 수출
x = pd.read_csv(EXT + r'\R2global_kr_export_hs6.csv', dtype={'hs6': str}, encoding='utf-8-sig'); x['hs6'] = x.hs6.str.zfill(6)
x = x[x.year <= 2016]
W = x.pivot_table(index='hs6', columns='year', values='value', aggfunc='sum')
bal = W.index[(W.reindex(columns=range(2000, 2017)) > 0).all(axis=1)]
link = mp.merge(pd.DataFrame({'hs6': bal}), on='hs6')
Xk = link.merge(x, on='hs6').groupby(['k5', 'year']).value.sum().unstack()
cov = {'ksic_mapped': mp.k5.nunique(), 'hs6_mapped': mp.hs6.nunique(), 'hs6_balanced_2000_2016': len(bal),
       'ksic_with_balanced_export': Xk.index.nunique()}
print('연계율', cov)
for ccs, nm in [(None, '5개국'), (['KR'], 'KR'), (['US', 'EP', 'JP'], 'US+EP+JP')]:
    P = patents(ccs)
    o, d = lead_test(Xk, P, range(2005, 2014), f'KR수출 vs 특허({nm})')
    res.append(o); print(o)
# --- 1b) 중국 수출 (5c 파일 재사용): 중국 수출 성장 vs 특허(CN) — 수출 2004–2018 매년 완결
a = pd.concat([pd.read_csv(EXT + r'\R2ind_comtrade_exports_5c_hs6_2004_2021.csv', dtype={'hs6': str}, usecols=['year','reporter','hs6','export_usd']),
               pd.read_csv(EXT + r'\R2N3_comtrade_exports_5c_hs6.csv', dtype={'hs6': str}, usecols=['year','reporter','hs6','export_usd'])])
a['hs6'] = a.hs6.str.zfill(6); a = a[(a.year <= 2016)].drop_duplicates(['year','reporter','hs6'])
for rep, cc in [(156, 'CN'), (842, 'US')]:
    b = a[a.reporter == rep]; Wb = b.pivot_table(index='hs6', columns='year', values='export_usd', aggfunc='sum')
    balb = Wb.index[(Wb.reindex(columns=range(2004, 2017)) > 0).all(axis=1)]
    Xb = mp.merge(pd.DataFrame({'hs6': balb}), on='hs6').merge(b, on='hs6').groupby(['k5','year']).export_usd.sum().unstack()
    for ccs, nm in [([cc], cc), (None, '5개국')]:
        o, d = lead_test(Xb, patents(ccs), range(2009, 2014), f'{cc}수출 vs 특허({nm})'); res.append(o); print(o)
# --- 2) ISO 표준
iso = pd.read_parquet(EXT + r'\trade_iso_ksic_year.parquet')
iso = iso[iso.year <= 2016]
act = iso[iso.year >= 2000].groupby('ksic').total.sum()
for thr in [20, 85]:
    ks = act[act >= thr].index
    for col in ['total', 'new']:
        S = iso[iso.ksic.isin(ks)].pivot_table(index='ksic', columns='year', values=col, aggfunc='sum').reindex(columns=range(1995, 2017)).fillna(0)
        o, d = lead_test(S, patents(None), range(2005, 2014), f'ISO표준({col}, KSIC 누적≥{thr}건) vs 특허(5개국)', x_log1p=True)
        o['n_ksic_linked'] = len(ks); res.append(o); print(o)
pd.DataFrame(res).to_csv(EXT + r'\trade_meaning_tests.csv', index=False, encoding='utf-8-sig')
json.dump(cov, open(EXT + r'\trade_export_link_coverage.json', 'w'), ensure_ascii=False)

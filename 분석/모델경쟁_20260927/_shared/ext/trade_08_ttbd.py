# -*- coding: utf-8 -*-
"""World Bank/Bown Temporary Trade Barriers Database(GAD, 2016판, 무료·무등록) → 한국 대상 반덤핑 조사개시 HS → KSIC×연도."""
import sys; sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext')
import zipfile, io, pandas as pd, numpy as np
from trade_lib import *
z = zipfile.ZipFile(EXT + r'\trade_ttbd_2016.zip'); rows = []
for n in [i for i in z.namelist() if i.startswith('GAD/GAD-') and i.endswith('.xls')]:
    x = pd.read_excel(io.BytesIO(z.read(n)), sheet_name=None, engine='xlrd', dtype=str)
    ms = [v for k, v in x.items() if k.endswith('Master')]; ps = [v for k, v in x.items() if k.endswith('Products')]
    if not ms or not ps: continue
    m = ms[0][['CASE_ID','INV_CTY_CODE','INIT_DATE','PRODUCT']]; p = ps[0][['CASE_ID','HS_CODE','HS_DIGITS']]
    d = m[m.INV_CTY_CODE == 'KOR'].merge(p, on='CASE_ID'); d['importer'] = n[8:11]; rows.append(d)
ad = pd.concat(rows)
ad['hs'] = ad.HS_CODE.fillna('').str.replace(r'[^0-9]', '', regex=True)
ad['year'] = pd.to_datetime(ad.INIT_DATE, errors='coerce').dt.year
ad = ad[(ad.hs.str.len() >= 4) & ad.year.notna()]
ad.to_parquet(EXT + r'\trade_ttbd_ad_vs_korea_hs.parquet', index=False)
print('한국 대상 AD 사건', ad.CASE_ID.nunique(), '수입국', ad.importer.nunique(), '연도', int(ad.year.min()), int(ad.year.max()))
mp = pd.read_csv(EXT.replace(r'모델경쟁_20260927\_shared\ext', r'fusion\ksic_hs6_map.csv'), dtype=str, encoding='utf-8-sig'); mp['hs6'] = mp.hs6.str.zfill(6)
mp['hs4'] = mp.hs6.str[:4]
a6 = ad[ad.hs.str.len() >= 6].assign(hs6=lambda d: d.hs.str[:6]).merge(mp[['k5','hs6']], on='hs6')
a4 = ad[ad.hs.str.len() == 4].assign(hs4=lambda d: d.hs).merge(mp[['k5','hs4']].drop_duplicates(), on='hs4')
L = pd.concat([a6[['CASE_ID','k5','year','importer']], a4[['CASE_ID','k5','year','importer']]]).drop_duplicates(['CASE_ID','k5'])
L.to_csv(EXT + r'\trade_ttbd_ad_vs_korea_ksic.csv', index=False, encoding='utf-8-sig')
print('KSIC 연결 사건', L.CASE_ID.nunique(), '/', ad.CASE_ID.nunique(), '| KSIC 수', L.k5.nunique(), '/50')
print(L.groupby('k5').CASE_ID.nunique().sort_values(ascending=False).head(15).to_string())
AD = L.groupby(['k5','year']).size().unstack(fill_value=0).reindex(columns=range(1990, 2017), fill_value=0)
AD = AD.reindex(patents().index, fill_value=0)
rows = []
for t in range(2005, 2014):
    r = pd.DataFrame({'ad': win(AD, t, -2, 0),
        'dP_fut': growth(patents(['KR']), t, 1, 3, -2, 0), 'dP_past': growth(patents(['KR']), t, -2, 0, -5, -3)})
    r['t'] = t; rows.append(r.reset_index())
d = pd.concat(rows).dropna(); d['ad_any'] = (d.ad > 0).astype(int)
import statsmodels.formula.api as smf
m = smf.ols('dP_fut ~ ad_any + dP_past + C(t)', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.ksic)[0]})
print('AD(t-2..t) → KR 특허성장 t+1..t+3: coef', round(m.params['ad_any'], 4), 'p', round(m.pvalues['ad_any'], 4), 'n', len(d), 'AD>0 관측', int(d.ad_any.sum()))
pd.DataFrame([dict(test='TTBD AD개시(t-2..t, 한국 대상) → KR 특허성장', coef=m.params['ad_any'], p=m.pvalues['ad_any'], n=len(d), n_ad=int(d.ad_any.sum()))]).to_csv(EXT + r'\trade_meaning_tests3.csv', index=False, encoding='utf-8-sig')

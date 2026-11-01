# -*- coding: utf-8 -*-
"""연계율 측정 (science_, 2026-09-28) — 로컬 계산만(제공 데이터 외부 전송 없음)
1) OpenAlex 키워드 타당도: US 특허 제목(영문) 중 해당 KSIC 키워드 포함 비율 vs 타 KSIC 포함 비율(lift)
2) OpenAlex 논문 수 충분성: 2005-2016 국가별 논문 >=30 인 KSIC 수
3) ANBERD 국가별 연결 KSIC 수(2005-2016 연속 관측), ECOS 연결 KSIC 수
"""
import os, re, duckdb, pandas as pd, numpy as np
from science_fetch_openalex import KW
from science_meaning_test import ISIC, ECOS, A2O, EXT, META
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
t = con.sql(f"select ksic, lower(title) title from read_parquet('{META}') where cc='US' and appl_year between 2005 and 2016 using sample 300000 rows").df()
rows = []
for k, kws in KW.items():
    pat = re.compile('|'.join(r'\b' + re.escape(w.lower()) for w in kws))
    m = t.title.str.contains(pat)
    rows.append({'ksic': k, 'hit_own': m[t.ksic == k].mean(), 'hit_other': m[t.ksic != k].mean(), 'n_own': int((t.ksic == k).sum())})
L = pd.DataFrame(rows); L['lift'] = L.hit_own / L.hit_other.replace(0, np.nan)
oa = pd.read_parquet(os.path.join(EXT, 'science_openalex_ksic_country_year.parquet'))
n = oa[oa.year.between(2005, 2016)].groupby(['ksic', 'country']).n_works.sum().unstack()
L = L.merge(n.add_prefix('papers0516_'), left_on='ksic', right_index=True)
an = pd.read_parquet(os.path.join(EXT, 'science_oecd_anberd_isic4.parquet'))
an = an[(an.CRITERIA == 'MA') & (an.UNIT_MEASURE == 'USD_PPP') & (an.PRICE_BASE == 'Q')]
for ra, o in A2O.items():
    ok = an[(an.REF_AREA == ra) & an.TIME_PERIOD.between(2005, 2016)].groupby('ACTIVITY').OBS_VALUE.count()
    L['anberd_' + o] = L.ksic.map(ISIC).map(lambda c: int(ok.get(c, 0)) >= 10)
ec = pd.read_parquet(os.path.join(EXT, 'science_ecos_rd_intensity.parquet'))
ok = ec[(ec['size'] == 'A') & (ec.stat_code == '501Y006')].groupby('ind_code').rd_to_sales_pct.count()
L['ecos'] = L.ksic.map(ECOS).map(lambda c: int(ok.get(c, 0)) >= 10)
L['isic'] = L.ksic.map(ISIC); L['ecos_code'] = L.ksic.map(ECOS)
L.to_csv(os.path.join(EXT, 'science_linkage_ksic50.csv'), index=False, encoding='utf-8-sig')
print(L.round(3).to_string())
print('lift>=3:', (L.lift >= 3).sum(), ' lift>=10:', (L.lift >= 10).sum(), ' median lift', L.lift.median())
for c in ['ALL', 'KR', 'US', 'JP', 'CN', 'DE']: print('papers>=30', c, (L['papers0516_' + c] >= 30).sum())
for o in A2O.values(): print('anberd', o, L['anberd_' + o].sum())
print('ecos', L.ecos.sum(), 'distinct anberd groups', L.isic.nunique(), 'distinct ecos groups', L.ecos_code.nunique())

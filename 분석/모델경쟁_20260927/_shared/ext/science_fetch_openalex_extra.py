# -*- coding: utf-8 -*-
"""OpenAlex 보조 신호 (science_ 트랙, 2026-09-28): 같은 KSIC 키워드 질의에
 (a) 기업 소속 저자 논문(authorships.institutions.type:company) ALL·KR  -> '기업 과학' 신호
 (b) 한국연구재단(NRF, OpenAlex funder F4320322120) 지원 논문 -> 정부 R&D 산출 대리지표
키 불필요·개인정보 미포함. 요청 간 1.2초. 출력 science_openalex_ksic_extra_year.parquet
"""
import os, time, pandas as pd
from science_fetch_openalex import KW, call
OUT = os.path.dirname(os.path.abspath(__file__))
VARS = {'company_ALL': 'authorships.institutions.type:company',
        'company_KR': 'authorships.institutions.type:company,authorships.countries:KR',
        'nrf_KR': 'funders.id:F4320322120'}
part = os.path.join(OUT, 'science_openalex_ksic_extra_year.parquet')
done = pd.read_parquet(part) if os.path.exists(part) else pd.DataFrame(columns=['ksic','variant','year','n_works'])
have = set(zip(done.ksic, done.variant)); rows = done.to_dict('records')
for k, kws in KW.items():
    q = '(' + ' OR '.join('"%s"' % w for w in kws) + ')'
    for v, f2 in VARS.items():
        if (k, v) in have: continue
        d, rem = call({'filter': 'title_and_abstract.search:%s,publication_year:1995-2025,%s' % (q, f2), 'group_by': 'publication_year'})
        got = {int(g['key']): g['count'] for g in d['group_by']}
        rows += [{'ksic': k, 'variant': v, 'year': y, 'n_works': got.get(y, 0)} for y in range(1995, 2026)]
        print(k, v, d['meta']['count'], 'remaining', rem, flush=True); time.sleep(1.2)
    pd.DataFrame(rows).to_parquet(part, index=False)
    if rem is not None and int(rem) < 300: print('budget guard stop'); break
for v, f2 in VARS.items():
    d, rem = call({'filter': 'publication_year:1995-2025,%s' % f2, 'group_by': 'publication_year'})
    rows += [{'ksic': 'TOTAL', 'variant': v, 'year': int(g['key']), 'n_works': g['count']} for g in d['group_by']]
    time.sleep(1.2)
pd.DataFrame(rows).to_parquet(part, index=False); print('done', len(rows))

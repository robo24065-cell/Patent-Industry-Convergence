# -*- coding: utf-8 -*-
r"""C18-T2 s1e: (a) D2 확정 기준(2021-04 전후 24개월 월평균 US 등록 건수 비) (b) 5개국 '절벽' 자동 탐지(PREREG [1](ii) 기준)
 (c) 같은 유형 추가 훑기: family 필드 안 토큰의 번호 형식 이중 표기(같은 국가 접두·숫자 정규화 키가 같고 원문이 다른 토큰 쌍) — T4·N8·N5·N4 등 family 토큰 사용처 노출 점검
 입력: cache/s1b_rgst_ym.csv, cache/s1b_fill_by_ay.csv(s1b_fields.py 산출), 제공 메타(읽기만) → cache/s1e.json, cache/s1e_cliffs.csv"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
import numpy as np, pandas as pd
R = {}
ym = pd.read_csv(os.path.join(C, 's1b_rgst_ym.csv'), dtype={'cc': str, 'ym': int})
ym = ym[(ym.ym >= 200001) & (ym.ym <= 202507)]
# (a) D2 기준
u = ym[ym.cc == 'US'].set_index('ym').n
months = pd.period_range('2000-01', '2025-07', freq='M'); idx = [int(p.strftime('%Y%m')) for p in months]
u = u.reindex(idx, fill_value=0)
pre = u[(u.index >= 201904) & (u.index <= 202103)].mean(); post = u[(u.index >= 202104) & (u.index <= 202303)].mean()
R['D2_us_pre24_mean'] = float(pre); R['D2_us_post24_mean'] = float(post); R['D2_ratio'] = float(pre / post); R['D2_defect_confirmed(<0.10)'] = bool(pre / post < 0.10)
# (b) 절벽: 인접 12개월 창 월평균 비 ≥5 (오름) 또는 ≤1/5 (내림) — 각 국가 등록 연월
rows = []
for cc, g in ym.groupby('cc'):
    s = g.set_index('ym').n.reindex(idx, fill_value=0).astype(float).values
    for i in range(12, len(s) - 11):
        a, b = s[i - 12:i].mean(), s[i:i + 12].mean()
        r = (b + 1) / (a + 1)
        if r >= 5 or r <= 0.2:
            rows.append(dict(kind='rgst_ym_12m', cc=cc, at=idx[i], before=round(a, 1), after=round(b, 1), ratio=round(r, 3)))
fa = pd.read_csv(os.path.join(C, 's1b_fill_by_ay.csv'))
fa = fa[(fa.appl_year >= 2000) & (fa.appl_year <= 2023)]
for cc, g in fa.groupby('cc'):
    g = g.sort_values('appl_year')
    for col in ['rgst_fill', 'rgstd_fill', 'pub_fill', 'ipc_fill', 'app_fill', 'fam_fill', 'prio_fill', 'title_fill']:
        d = g[col].diff()
        for yr, dv, v in zip(g.appl_year, d, g[col]):
            if pd.notna(dv) and abs(dv) >= 0.3:
                rows.append(dict(kind='fill_by_appl_year', cc=cc, at=int(yr), before=round(v - dv, 3), after=round(v, 3), ratio=round(dv, 3), field=col))
cl = pd.DataFrame(rows)
# 연속 창은 첫 지점만 남겨 요약(같은 절벽이 12개월 안에서 반복 표시되는 것 정리)
cl.to_csv(os.path.join(C, 's1e_cliffs.csv'), index=False, encoding='utf-8-sig')
summ = []
for (k, cc, fld), g in cl.groupby(['kind', 'cc', cl.get('field', pd.Series(['-'] * len(cl))).fillna('-')]):
    g = g.sort_values('at'); prev = None
    for _, r in g.iterrows():
        if k == 'rgst_ym_12m':
            p = pd.Period(str(r['at']), 'M')
            if prev is not None and (p - prev).n <= 12: prev = p; continue
            prev = p
        summ.append(dict(kind=k, cc=cc, field=fld, at=int(r['at']), before=r['before'], after=r['after'], ratio=r['ratio']))
R['cliffs_summary'] = summ
# (c) family 토큰 번호 형식 이중 표기
con = con3()
con.execute(f"""create table tk as select cc::varchar cc, appl_no, unnest(string_split(family,'|')) tok from read_parquet('{META}') where family is not null and family<>''""")
con.execute(r"""create table tk2 as select cc, appl_no, tok, regexp_extract(tok,'^([A-Z]{2})',1) pfx, ltrim(regexp_replace(substr(tok,3),'[^0-9]','','g'),'0') k from tk where length(tok)>2""")
R['family_tokens'] = con.sql("select count(*), count(distinct cc||appl_no) from tk2").fetchone()
d = con.sql("""select cc, pfx, count(*) n_pairs, count(distinct cc||appl_no) n_records from (
     select cc, appl_no, pfx, k, count(distinct tok) c from tk2 where k<>'' group by all having c>1) group by 1,2 order by n_pairs desc""").df()
R['family_token_double_notation'] = d.head(40).values.tolist()
ex = con.sql("""select cc, appl_no, pfx, k, list(distinct tok) toks from tk2 where k<>'' group by all having count(distinct tok)>1 using sample 40 rows""").df()
ex.to_csv(os.path.join(C, 's1e_family_token_examples.csv'), index=False, encoding='utf-8-sig')
# KR 기록의 US 토큰만(T4 fam_us·N8 fam_us·fam_n 의 원천): 길이 분포와 이중 표기 기록 수
R['kr_us_token_len'] = con.sql("select length(tok), count(*) from tk2 where cc='KR' and pfx='US' group by 1 order by 1").fetchall()
R['kr_records_with_us_double'] = con.sql("""select count(distinct appl_no) from (select appl_no, k, count(distinct tok) c from tk2 where cc='KR' and pfx='US' and k<>'' group by all having c>1)""").fetchone()[0]
R['kr_records_with_family'] = con.sql("select count(distinct appl_no) from tk2 where cc='KR'").fetchone()[0]
jdump(R, 's1e.json'); out(json.dumps({k: v for k, v in R.items() if k != 'cliffs_summary'}, ensure_ascii=False, default=str)[:3000])
out(pd.DataFrame(summ).to_string())

# -*- coding: utf-8 -*-
"""verify_0c: 수정 연결(V2 같은 우선일, V1 초대형 제거)이 수동 판정 '다른 발명' 쌍을 실제로 걸러내는지 확인(민감도의 타당성 반박 시도).
읽기만: res/s2_manual100.csv, cache/manual_sheet.csv, cache/fam_us_V1/V2.parquet, _shared fam_us.parquet. 쓰기: verify_0/v0c_out.json"""
import json, sys
sys.dont_write_bytecode = True
import pandas as pd, numpy as np
B = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
W = B + "/분석/검증루프_20261012/C11/C11-T3"
EXT = B + "/분석/모델경쟁_20260927/_shared/ext"
import duckdb
c = duckdb.connect(); c.execute("set threads=2; set memory_limit='2GB'")
# s2b_manual_sample.py 와 같은 표본 재구성(id = cc, appl_no 순 row_number), manual_sheet 의 us_pn·k_appl 과 대조
s = c.sql(f"""with s as (select * from read_parquet('{W}/cache/s2_fin.parquet') where kind='link' and t3win and cc<>'US' and n_u>0
   order by hash(appl_no||us_pn::varchar||'manual20261013') limit 100)
   select row_number() over (order by s.cc, s.appl_no) id, s.cc, s.appl_no k_appl_no, s.us_pn, r.appl_date from s join read_parquet('{W}/cache/rec.parquet') r using(cc, appl_no)""").df()
sh = pd.read_csv(W + "/cache/manual_sheet.csv", dtype={'k_appl': str})
chk = s.merge(sh[['id', 'us_pn', 'k_appl']], on='id', suffixes=('', '_sh'))
assert len(chk) == 100 and (chk.us_pn == chk.us_pn_sh).all() and (chk.appl_date == chk.k_appl).all(), 'sample mismatch'
j = pd.read_csv(W + "/res/s2_manual100.csv")
m = j.merge(s[['id', 'k_appl_no']], on='id').rename(columns={'k_appl_no': 'k_appl'}); m['k_appl'] = m.k_appl.astype(str)
out = {'judge_counts': m.judge.value_counts().to_dict()}
F0 = pd.read_parquet(EXT + "/r3ros_work/fam_us.parquet", columns=['cc', 'appl_no', 'us_pn', 'src'])
F0['appl_no'] = F0.appl_no.astype(str)
key = m[['cc', 'k_appl', 'us_pn']].rename(columns={'k_appl': 'appl_no'})
for V in ('V0', 'V1', 'V2'):
    if V == 'V0': F = F0
    else:
        F = pd.read_parquet(W + f"/cache/fam_us_{V}.parquet", columns=['cc', 'appl_no', 'us_pn']); F['appl_no'] = F.appl_no.astype(str)
    hit = key.merge(F[['cc', 'appl_no', 'us_pn']].drop_duplicates(), on=['cc', 'appl_no', 'us_pn'], how='left', indicator=True)
    m[f'kept_{V}'] = (hit['_merge'] == 'both').values
out['kept_by_judge'] = {V: m.groupby('judge')[f'kept_{V}'].agg(['sum', 'count']).to_dict('index') for V in ('V0', 'V1', 'V2')}
for V in ('V1', 'V2'):
    k = m[m[f'kept_{V}'] & m.kept_V0]
    out[f'same_inv_share_after_{V}'] = dict(n_kept=int(len(k)), same=int(k.same_invention.sum()), share=float(k.same_invention.mean()),
                                         dropped_diff=int((~m[f'kept_{V}'] & m.kept_V0 & (m.same_invention == 0) & (m.judge != 'U')).sum()),
                                         dropped_same=int((~m[f'kept_{V}'] & m.kept_V0 & (m.same_invention == 1)).sum()))
json.dump(out, open(W + "/verify_0/v0c_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

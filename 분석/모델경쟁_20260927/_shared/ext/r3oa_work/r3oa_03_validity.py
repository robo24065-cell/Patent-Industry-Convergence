# -*- coding: utf-8 -*-
"""r3oa 03: 검색어 타당도(로컬 계산, 외부 전송 없음).
US 출원(2005–2016) 영문 제목에서 메인그룹별 검색어 적중률: 자기 메인그룹 vs 다른 메인그룹.
(OpenAlex는 제목+초록·어간 검색이라 실제 적중은 더 넓다. 여기서는 상대 비교만.)"""
import duckdb, pandas as pd, numpy as np, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3oa_terms import TERMS
OUT=os.path.dirname(os.path.abspath(__file__))
BASE=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R2\cache_dev\base_rows.parquet'
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
print(con.sql(f"describe select * from read_parquet('{BASE}')").fetchall())
U=con.sql(f"""select distinct appl_no, mgn(ipc_raw) mg, lower(title) t from read_parquet('{BASE}')
   where cc='US' and year between 2005 and 2016 and title is not null""").df()
U=U.dropna(subset=['mg'])
print('US apps rows',len(U))
keep=set(TERMS)
U['in']=U.mg.isin(keep)
rows=[]
def norm(w):
    w=w.lower(); return w
for mg,ts in TERMS.items():
    pat='|'.join(re.escape(norm(w)) for w in ts)
    hit=U.t.str.contains(pat,regex=True)
    self_m=U.mg==mg
    n_self=int(self_m.sum())
    r_self=float(hit[self_m].mean()) if n_self else np.nan
    r_oth=float(hit[~self_m & U['in']].mean())
    # 자기 메인그룹의 적중 순위(메인그룹별 적중률, n>=30인 메인그룹 중)
    g=hit.groupby(U.mg).agg(['mean','size']); g=g[g['size']>=30]
    rank=int((g['mean']>r_self).sum()+1) if n_self>=30 else np.nan
    share_hits_self=float(hit[self_m].sum()/max(hit.sum(),1))
    rows.append(dict(mg=mg,n_us_titles=n_self,hit_self=r_self,hit_other=r_oth,lift=r_self/r_oth if r_oth>0 else np.nan,
                     self_rank=rank,n_mg_ranked=len(g),share_of_hits_in_self=share_hits_self))
V=pd.DataFrame(rows)
V.to_csv(os.path.join(OUT,'r3oa_query_validity_v1.csv'),index=False)
print(V.describe().T)
print((V.hit_self<0.02).sum(),'mg with self-hit <2%')
print(V.sort_values('hit_self').head(60).to_string())

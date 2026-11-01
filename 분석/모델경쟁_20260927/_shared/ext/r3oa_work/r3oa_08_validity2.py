# -*- coding: utf-8 -*-
"""r3oa 08 (03을 보강분+전체로 재계산, 2026-09-29 재시도): 검색어 타당도(로컬 계산, 외부 전송 없음).
US 출원(2005–2016) 영문 제목에서 메인그룹별 검색어 적중률: 자기 메인그룹 vs 다른 메인그룹.
(OpenAlex는 제목+초록·어간 검색이라 실제 적중은 더 넓다. 여기서는 상대 비교만.)"""
import duckdb, pandas as pd, numpy as np, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3oa_terms import TERMS
from r3oa_terms_b import TERMS_B
TB=set(TERMS_B); TERMS=dict(TERMS); TERMS.update(TERMS_B)
OUT=os.path.dirname(os.path.abspath(__file__))
BASE=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측\R2\cache_dev\base_rows.parquet'
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
U=con.sql(f"""select distinct appl_no, mgn(ipc_raw) mg, lower(title) t from read_parquet('{BASE}')
   where cc='US' and year between 2005 and 2016 and title is not null""").df()
U=U.dropna(subset=['mg'])
print('US apps rows',len(U))
keep=set(TERMS)
U['in']=U.mg.isin(keep).values
codes,uniq=pd.factorize(U.mg); cnt=np.bincount(codes); ok=cnt>=30
inarr=U['in'].values
rows=[]
for mg,ts in TERMS.items():
    if mg not in TB: continue   # A셋은 v1 결과 재사용(순위 기준 모집단 동일)
    pat='|'.join(re.escape(w.lower()) for w in ts)
    hit=U.t.str.contains(pat,regex=True).values
    k=np.where(uniq==mg)[0]; n_self=int(cnt[k[0]]) if len(k) else 0
    hc=np.bincount(codes,weights=hit,minlength=len(uniq)); rate=hc/np.maximum(cnt,1)
    r_self=float(rate[k[0]]) if n_self else np.nan
    oth=inarr & (codes!=(k[0] if len(k) else -1)); r_oth=float(hit[oth].mean())
    rank=int((rate[ok]>r_self).sum()+1) if n_self>=30 else np.nan
    rows.append(dict(mg=mg,set='B',n_us_titles=n_self,hit_self=r_self,hit_other=r_oth,lift=r_self/r_oth if r_oth>0 else np.nan,
                     self_rank=rank,n_mg_ranked=int(ok.sum()),share_of_hits_in_self=float(hc[k[0]]/max(hit.sum(),1)) if n_self else 0.0))
V1=pd.read_csv(os.path.join(OUT,'r3oa_query_validity_v1.csv')); V1.insert(1,'set','A')
rows=V1.to_dict('records')+rows
V=pd.DataFrame(rows)
V.to_csv(os.path.join(OUT,'r3oa_query_validity_v2.csv'),index=False)
for k,g in V.groupby('set'):
    h=g[g.n_us_titles>=30]; r=g.dropna(subset=['self_rank']); r=r[r.hit_self>0]
    print(k,'n',len(g),'n>=30',len(h),'self med %.4f mean %.4f other mean %.5f'%(h.hit_self.median(),h.hit_self.mean(),h.hit_other.mean()),
      '>=1%% %.3f >=5%% %.3f'%((h.hit_self>=0.01).mean(),(h.hit_self>=0.05).mean()),
      'rank n',len(r),'med',r.self_rank.median(),'top1 %.3f top5 %.3f top10 %.3f'%((r.self_rank==1).mean(),(r.self_rank<=5).mean(),(r.self_rank<=10).mean()),'n_mg_ranked',int(g.n_mg_ranked.iloc[0]))
print(V[V.set=='B'].sort_values('hit_self').head(20).to_string())

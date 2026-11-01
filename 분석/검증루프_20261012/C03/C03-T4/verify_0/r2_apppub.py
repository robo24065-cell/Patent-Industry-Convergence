# 반박 검증 r2: T1 app_pub 구조(등록 문서도 코드가 있는지), 연결 vs 비연결의 등록여부 SMD(사전 속성, 미보고분)
import sys, duckdb, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
AP = B + '/분석/모델경쟁_20260927/T1_산업대기실/cache/app_pub.parquet'
KRP = B + '/공모전데이터/작업데이터/kr_patents.parquet'
DOCS = B + '/분석/검증루프_20261012/C02/C02-T4/cache/docs.parquet'
PAT = B + '/분석/검증루프_20261012/C02/C02-T4/cache/pat.parquet'
d = duckdb.connect(); d.execute("set memory_limit='3GB'; set threads=2")
print(d.sql(f"describe select * from read_parquet('{AP}')").df()[['column_name','column_type']].to_string())
print(d.sql(f"select reg, count(*), count(distinct appl_no), avg((applicant_cd is not null)::int) from read_parquet('{AP}') group by 1").fetchall())
# 연결(any) vs 비연결: 등록여부 SMD (개발 공개 2010–2024)
q = f"""with kp as (select distinct 출원번호 a, 출원인명 an, 우선권국가 pr from read_parquet('{KRP}') where length(공개일자)=8 and 공개일자 between '20100101' and '20241231'),
 rg as (select distinct appl_no a from read_parquet('{PAT}')),
 dc as (select appl_no a, max((substr(cd,1,1)='1')::int) corp from read_parquet('{DOCS}') group by 1)
 select (dc.a is not null) linked, count(*) n, avg((rg.a is not null)::int) reg,
   avg(case when regexp_matches(coalesce(an,''),'주식회사|\(주\)|㈜|유한회사') and (coalesce(pr,'')='' or pr like '%KR%') then 1 else 0 end) corpname_dom
 from kp left join rg using(a) left join dc using(a) group by 1"""
r = d.sql(q).df(); print(r.to_string())
p1 = r[r.linked].reg.iloc[0]; p0 = r[~r.linked].reg.iloc[0]
print('SMD reg (any link)', round((p1-p0)/np.sqrt((p1*(1-p1)+p0*(1-p0))/2), 3))
q2 = f"""with kp as (select distinct 출원번호 a from read_parquet('{KRP}') where length(공개일자)=8 and 공개일자 between '20100101' and '20241231'
          and regexp_matches(coalesce(출원인명,''),'주식회사|\(주\)|㈜|유한회사|corporation|co\.|ltd|inc','i') and not (coalesce(우선권국가,'')<>'' and 우선권국가 not like '%KR%')),
 rg as (select distinct appl_no a from read_parquet('{PAT}')),
 dc as (select distinct appl_no a from read_parquet('{DOCS}') where substr(cd,1,1)='1')
 select (dc.a is not null) linked, count(*) n, avg((rg.a is not null)::int) reg from kp left join rg using(a) left join dc using(a) group by 1"""
r2 = d.sql(q2).df(); print(r2.to_string())
p1 = r2[r2.linked].reg.iloc[0]; p0 = r2[~r2.linked].reg.iloc[0]
print('SMD reg (corp link)', round((p1-p0)/np.sqrt((p1*(1-p1)+p0*(1-p0))/2), 3))

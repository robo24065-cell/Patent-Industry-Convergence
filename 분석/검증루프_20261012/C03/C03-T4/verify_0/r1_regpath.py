# 반박 검증 r1: 문서-기업코드 연결이 등록 여부(2026 스냅샷 기준, 사후 정보)에 따라 달라지는지. 개발 구간(공개<2025)만 단위 수준, 2025-H1은 건수만.
import sys, duckdb
sys.stdout.reconfigure(encoding='utf-8')
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
KRP = B + '/공모전데이터/작업데이터/kr_patents.parquet'
DOCS = B + '/분석/검증루프_20261012/C02/C02-T4/cache/docs.parquet'
PAT = B + '/분석/검증루프_20261012/C02/C02-T4/cache/pat.parquet'
d = duckdb.connect(); d.execute("set memory_limit='3GB'; set threads=2")
print(d.sql(f"describe select * from read_parquet('{KRP}')").df()[['column_name','column_type']].to_string())
print(d.sql(f"select count(*) n, count(distinct 출원번호) na from read_parquet('{KRP}')").fetchall())
print('dup appl with diff ksic', d.sql(f"select count(*) from (select 출원번호 from read_parquet('{KRP}') group by 1 having count(distinct 세세분류코드)>1)").fetchall())
print(d.sql(f"select reg, count(*), count(distinct appl_no) from read_parquet('{DOCS}') group by 1").fetchall())
# 공개 문서별: 등록원부 patent 존재(등록됨) 여부, 연결 경로
q = f"""with kp as (select distinct 출원번호 a, substr(공개일자,1,4) py, substr(공개일자,1,6) ym from read_parquet('{KRP}') where length(공개일자)=8 and 공개일자 < '20250701'),
 rg as (select distinct appl_no a from read_parquet('{PAT}')),
 dc as (select appl_no a, max(reg::int) creg, max((substr(cd,1,1)='1')::int) corp from read_parquet('{DOCS}') group by 1)
 select case when ym>='202501' then '2025H1' else py end yy, count(*) n,
   avg((rg.a is not null)::int) reg_rate,
   avg((dc.a is not null)::int) link_any,
   avg(case when rg.a is not null then (dc.a is not null)::int end) link_if_reg,
   avg(case when rg.a is null then (dc.a is not null)::int end) link_if_unreg,
   avg(coalesce(dc.corp,0)) corp_doc_rate,
   avg(case when dc.corp=1 then dc.creg end) corpdoc_via_reg
 from kp left join rg using(a) left join dc using(a) where py>='2010' group by 1 order by 1"""
print(d.sql(q).df().round(4).to_string())

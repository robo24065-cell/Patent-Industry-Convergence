# s0c: 제공 메타 해외 문서의 우선권·패밀리 필드 형식 탐색(2025 이후 공개는 연도별 count만)
import sys, duckdb
sys.stdout.reconfigure(encoding='utf-8')
M = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
d = duckdb.connect(); d.execute("set memory_limit='3GB'; set threads=2")
d.execute(f"create view m as select * from read_parquet('{M}')")
print(d.sql("select cc, count(*) n, count(distinct appl_no) na, avg((family is not null and family<>'')::int) fam, avg((prio_cc like '%KR%')::int) prkr from m group by cc order by cc").fetchall())
print(d.sql("select cc, substr(pub_date,1,4) y, count(distinct appl_no) from m where cc in ('US','EP','CN') and prio_cc like '%KR%' and pub_date>='2018' group by 1,2 order by 1,2").fetchall())
for cc in ['US', 'EP', 'CN']:
    print(cc, d.sql(f"select appl_no, pub_date, prio_cc, prio_date, family, applicant, ksic from m where cc='{cc}' and prio_cc like '%KR%' and pub_date<'2020' limit 4").fetchall())
print('KR sample', d.sql("select appl_no, pub_date, prio_cc, family, applicant from m where cc='KR' and family is not null and family<>'' and pub_date<'2020' limit 4").fetchall())

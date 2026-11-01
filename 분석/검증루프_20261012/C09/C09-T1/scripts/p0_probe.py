# -*- coding: utf-8 -*-
# 탐색 probe (판정 미사용): 현대자동차 이름 변형과 연도별 제공 범위 문서 수
import sys, duckdb
sys.stdout.reconfigure(encoding="utf-8")
META = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet"
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
print(con.sql(f"""select applicant, count(distinct appl_no) n, min(appl_year), max(appl_year) from read_parquet('{META}')
 where cc='KR' and (applicant like '%현대자동차%' or applicant like '%기아%' or applicant ilike '%hyundai motor%') group by 1 order by 2 desc limit 25""").df().to_string())
print(con.sql(f"""select appl_year, count(distinct appl_no) n, count(distinct case when rgst_no is not null and rgst_no<>'' then appl_no end) nreg from read_parquet('{META}')
 where cc='KR' and applicant='현대자동차주식회사' group by 1 order by 1""").df().to_string())
print(con.sql(f"""select appl_year, count(distinct appl_no) n from read_parquet('{META}')
 where cc='KR' and applicant='기아 주식회사' group by 1 order by 1""").df().to_string())
print(con.sql(f"""select ksic, appl_year, count(distinct appl_no) n from read_parquet('{META}')
 where cc='KR' and applicant='현대자동차주식회사' and appl_year between 2013 and 2019 group by 1,2 order by 3 desc limit 30""").df().to_string())
print(con.sql(f"""select substr(main_ipc,1,4) sc, appl_year, count(distinct appl_no) n from read_parquet('{META}')
 where cc='KR' and applicant='현대자동차주식회사' and appl_year between 2014 and 2018 group by 1,2 order by 3 desc limit 30""").df().to_string())

import duckdb, sys
sys.stdout.reconfigure(encoding='utf-8')
f = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext\ntis_patent_link.parquet'.replace('\\', '/')
print(duckdb.sql(f"select PerformAgent_code, PerformAgent, perform_grp, count(*) n from read_parquet('{f}') group by all order by 1").df().to_string())

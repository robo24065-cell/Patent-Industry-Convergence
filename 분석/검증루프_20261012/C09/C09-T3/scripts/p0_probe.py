"""p0: base_rows 스키마·이름 표본·결측/법인형 비율 탐색(읽기 전용)."""
import sys, os, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
B = ddb(os.path.join(CP_FIN, 'base_rows.parquet'))
print(con.sql(f"describe select * from read_parquet('{B}')").df()[['column_name', 'column_type']].to_string())
for cc in ['US', 'KR', 'JP', 'CN', 'EP']:
    print('==', cc)
    print(con.sql(f"select appl_year, year, app1 from read_parquet('{B}') where cc='{cc}' and app1 is not null and app1<>'' using sample 12 rows").df().to_string())
for cc, rg in CORP_RE.items():
    q = f"""select appl_year, count(*) n, avg((app1 is null or app1='')::int) miss,
            avg(case when app1 is null or app1='' then null else regexp_matches(app1, '{rg.replace("'", "''")}')::int end) corp
            from read_parquet('{B}') where cc='{cc}' and appl_year between 2000 and 2024 group by 1 order by 1"""
    print('==', cc); print(con.sql(q).df().round(3).T.to_string())

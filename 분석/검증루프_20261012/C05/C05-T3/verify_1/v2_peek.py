# -*- coding: utf-8 -*-
import duckdb, pandas as pd, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
M = (B + r"\분석\모델경쟁_20260927\_shared\meta\*.parquet").replace(os.sep, "/")
print(duckdb.sql(f"select appl_no, applicant, rgst_no from read_parquet('{M}') where cc='KR' and appl_year=2023 limit 5").df().to_string())
K = (B + r"\공모전데이터\작업데이터\kr_patents.parquet").replace(os.sep, "/")
print(duckdb.sql(f"describe select * from read_parquet('{K}')").df()[['column_name', 'column_type']].to_string())
print(duckdb.sql(f"select * from read_parquet('{K}') where 출원일 >= '20230101' limit 2").df().T.to_string())

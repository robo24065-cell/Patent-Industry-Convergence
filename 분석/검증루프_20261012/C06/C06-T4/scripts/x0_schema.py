import sys; sys.stdout.reconfigure(encoding='utf-8')
import pyarrow.parquet as pq
B=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
for p in [B+r'\공모전데이터\작업데이터\kr_patents.parquet', B+r'\공모전데이터\작업데이터\country_year_cube.parquet', B+r'\분석\모델경쟁_20260927\_ideas_R8\cache\kr_assign.parquet']:
    f=pq.ParquetFile(p); print(p[-40:], f.metadata.num_rows, f.schema_arrow.names)
v=pq.read_table(r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T4\cache\venture_clean.parquet').to_pandas()
print(v.region.value_counts().to_dict()); print(v.vtype.value_counts().to_dict())

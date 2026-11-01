import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import pyarrow.parquet as pq
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
E = os.path.join(B, '분석', '모델경쟁_20260927', '_shared', 'ext')
for f in ['firm_nps_wkpl_202412.parquet', 'firm_nps_wkpl_201512.parquet', 'ntis_project_raw.parquet', 'ntis_patent_flag.parquet']:
    m = pq.ParquetFile(os.path.join(E, f)); print(f, m.metadata.num_rows, m.schema_arrow.names)
for f in [os.path.join(B, '분석', '검증루프_20261012', 'C02', 'C02-T4', 'cache', 'kr_publab_src.parquet'),
          os.path.join(B, '공모전데이터', '작업데이터', 'country_year_cube.parquet'),
          os.path.join(B, '공모전데이터', '작업데이터', 'kr_patents.parquet')]:
    m = pq.ParquetFile(f); print(os.path.basename(f), m.metadata.num_rows, m.schema_arrow.names)
import pandas as pd
d = pd.read_parquet(os.path.join(E, 'firm_nps_wkpl_202412.parquet'), columns=None).head(3)
print(d.T)

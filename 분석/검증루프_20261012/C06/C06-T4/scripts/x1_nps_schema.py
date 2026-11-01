import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, pyarrow.parquet as pq
E=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
for s in ['201512','202012','202412']:
    f=pq.ParquetFile(E+rf'\firm_nps_wkpl_{s}.parquet'); print(s, f.metadata.num_rows, f.schema_arrow.names)
    d=f.read_row_group(0).to_pandas().head(3); print(d[['ym','biz6','stat','form','adpt_dt','n_emp']].to_string())

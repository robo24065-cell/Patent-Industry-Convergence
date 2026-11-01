import sys; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import pyarrow.parquet as pq
f=pq.ParquetFile(EXT+r"\ntis_project_raw.parquet"); print(f.metadata.num_rows); print(f.schema_arrow.names)
d=pq.ParquetFile(MC+r"\N7_특허기업_정부RnD첫수행\cache\dev_panel.parquet"); print(d.metadata.num_rows, d.schema_arrow.names[:14])
d=pq.ParquetFile(MC+r"\N7_특허기업_정부RnD첫수행\cache\final_panel.parquet"); print(d.metadata.num_rows, d.schema_arrow.names[:14])

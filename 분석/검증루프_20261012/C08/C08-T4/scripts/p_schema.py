# C08-T4 스키마 확인(읽기 전용)
import sys, os, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
fs = [os.path.join(B, '공모전데이터', '작업데이터', f) for f in ['kr_patents.parquet', 'kr_registered.parquet', 'country_year_cube.parquet']]
meta = os.path.join(B, '분석', '모델경쟁_20260927', '_shared', 'meta')
if os.path.isdir(meta):
    for f in sorted(os.listdir(meta)):
        fs.append(os.path.join(meta, f))
for f in fs:
    if f.endswith('.parquet'):
        m = pq.ParquetFile(f)
        print('==', os.path.basename(f), m.metadata.num_rows)
        print(', '.join(f'{x.name}:{x.type}' for x in m.schema_arrow))
    else:
        print('==', f, os.path.getsize(f))

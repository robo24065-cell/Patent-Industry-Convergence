import sys, os, json, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
R3 = os.path.join(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T3_기술셀부상예측", "R3")
for f in ['FINAL_R3_t0_2021', 'dev_preds_r3']:
    d = pd.read_parquet(os.path.join(R3, 'preds', f + '.parquet')); print(f, d.shape); print(d.head(3).to_string()); print(d.dtypes.to_dict())
    if 't0' in d: print(d.groupby('t0').size().to_dict())
d = json.load(open(os.path.join(R3, 'dev_result_r3.json'), encoding='utf-8')); print(json.dumps(d, ensure_ascii=False)[:2500])
import pyarrow.parquet as pq
pf = pq.ParquetFile(os.path.join(R3, 'cache_pub_final', 'base_rows.parquet')); print(pf.metadata.num_rows, pf.schema_arrow)

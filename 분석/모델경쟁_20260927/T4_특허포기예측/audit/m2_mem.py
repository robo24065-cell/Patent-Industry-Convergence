import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, xgboost as xgb, pyarrow.parquet as pq
from final_model import peak_mem_gb
def m(t): print(t, 'peak', peak_mem_gb(), flush=True)
sp = json.load(open('spec_v52.json', encoding='utf-8')); cfg = sp['components']['haz']; CAT = sp['categorical']
m('start')
t = pq.read_table('final_models/run_v52/frame_haz.parquet', columns=cfg['num'] + CAT + ['y', 'L']); m('table')
print(t.schema.types[:3], t.nbytes/1e9)
d = t.to_pandas(strings_to_categorical=True, self_destruct=True); del t; m('pandas')
print(d.dtypes.value_counts())
y = d.y.values.astype('float32'); d = d.drop(columns=['y', 'L'])
for c in d.columns:
    if c not in CAT and d[c].dtype != np.float32: d[c] = d[c].astype('float32')
gc.collect(); m('float32')
X = d[cfg['num'] + CAT]; m('X select')
dm = xgb.QuantileDMatrix(X, y, enable_categorical=True, max_bin=256); m('qdm')
del X; gc.collect()
dm2 = xgb.QuantileDMatrix(d[cfg['num']].to_numpy(), y, max_bin=256); m('qdm numpy no cats')

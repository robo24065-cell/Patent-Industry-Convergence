import os, sys, json, gc
os.environ['OMP_NUM_THREADS']='2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, xgboost as xgb
from final_model import peak_mem_gb
import ctypes
pr = None
def m(t): print(t, 'peak', peak_mem_gb(), flush=True)
sp = json.load(open('spec_v52.json', encoding='utf-8')); cfg = sp['components']['haz']; CAT = sp['categorical']
m('start')
d = pd.read_parquet('final_models/run_v52/frame_haz.parquet', columns=cfg['num'] + CAT + ['y', 'L']); m('read')
d = d.astype({c: 'category' for c in CAT}); gc.collect(); m('cat')
cmap = {c: sorted(d[c].astype(str).unique()) for c in CAT}
X = d[cfg['num']].astype('float32').copy()
for c in CAT: X[c] = pd.Categorical(d[c].astype(str), categories=cmap[c])
m('xmat')
y = d.y.values.astype('float32')
dm = xgb.QuantileDMatrix(X, y, enable_categorical=True, max_bin=256); m('qdm')
del X; gc.collect(); m('delX')
P = dict(cfg['params']); b = xgb.train(P, dm, 20); m('train20')

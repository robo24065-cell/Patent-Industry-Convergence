import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r3lib, r3ext, numpy as np, pandas as pd
R = r3lib.R
t=time.time()
linked, P = r3ext._nps(); print('load', time.time()-t, flush=True)
cur = P['2017-12']; print(cur.index.is_unique, cur.index.dtype, len(cur), cur.dtypes.to_dict(), flush=True)
d = R._read_frame('O18', 'ev.parquet', ['acd'])
a = d.acd.astype(str).values[:240000]
t=time.time(); lk = np.array([x in linked for x in a]); print('lk', time.time()-t, flush=True)
t=time.time(); inn = np.isin(a, cur.index.values); print('isin', time.time()-t, flush=True)
t=time.time(); e = cur.n_emp.reindex(a).values; print('reindex', time.time()-t, flush=True)

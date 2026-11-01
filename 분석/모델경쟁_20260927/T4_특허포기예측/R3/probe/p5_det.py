import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r3lib; R = r3lib.R
base = os.path.join(r3lib.R3, 'probe', 'det')
R.odir = lambda o: os.path.join(base, o)
spec = json.load(open(os.path.join(r3lib.R3, 'specs', 'E00.json'), encoding='utf-8'))['spec']
fn = R.fit_predict_xgb_bag('O12', spec, 'haz', 1)
a = np.load(fn); key = R.comp_key(spec, 'haz')
b = np.load(os.path.join(r3lib.R3, 'runs', 'O12', 'pred', f'{key}_b1.npy'))
print('max abs diff', float(np.abs(a - b).max()), 'corr', float(np.corrcoef(a, b)[0, 1]))

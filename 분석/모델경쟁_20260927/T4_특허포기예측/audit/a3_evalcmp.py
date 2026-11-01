import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import t4lib as T, soc as SO, feat2 as F2
from versions import add_ratio
sp = json.load(open('spec_v45.json', encoding='utf-8'))
num = sp['components']['cls']['num']; cats = sp['categorical']
L = '2014-01-01'
f = T.features(L); f = f.assign(acd=T.load()['pt'].acd.values[f.pid.values])
d = add_ratio(f); d = SO.add_soc(d); d = F2.add_fam_dated(d); d = F2.fix_dspst(d); d = SO.add_ak(d, 'next_annl'); d = F2.add_macro(d); d = F2.add_agent(d); d = F2.add_owner(d)
b = pd.read_parquet('cache/s2_ev.parquet'); b = b[b.L == L].reset_index(drop=True)
F = ['fam_n', 'fam_cc', 'fam_us', 'fam_ep', 'fam_jp', 'fam_cn', 'fam_wo']
for c in F: b[c] = b['fd_' + c]
b['dspst'] = b['dspst_fix']
print(len(d), len(b), (d.pid.values == b.pid.values).mean())
for c in num + cats:
    x, y = d[c], b[c]
    if c in cats: diff = (x.astype(str).values != y.astype(str).values).mean()
    else: diff = (~np.isclose(x.astype(float).values, y.astype(float).values, equal_nan=True, rtol=1e-4)).mean()
    if diff > 0: print(c, round(diff, 5))
a = pd.read_parquet('final_models/run_v45/pred_20140101.parquet'); p = pd.read_parquet('preds/pred_s2_ow_mac.parquet'); p = p[p.L == L]
print('corr p_cls vs s2 ow_mac', np.corrcoef(a.p_cls, p.p)[0, 1])

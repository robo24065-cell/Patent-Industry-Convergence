"""s3b: all_newapp_sh 의 시나리오별(S0·S1·S2·S2b) 원점 대 최종(2021) PSI — 분포 이동의 원인 분해 → cache/s3b_psi_by_origin_scen.csv"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
SF = pd.read_parquet(os.path.join(CACHE, 'scen_feats_fin.parquet'))
SF = SF[SF.tot_W_m >= 50]
def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); q[0] = -np.inf; q[-1] = np.inf
    pa = np.clip(np.histogram(a, q)[0] / len(a), 1e-4, None); pb = np.clip(np.histogram(b, q)[0] / len(b), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))
cols = ['all_newapp_sh_S0', 'all_newapp_sh_S1', 'all_newapp_sh_S2', 'all_newapp_sh_S2b']
rows = []
fin = SF[SF.t0 == 2021]
for t0 in range(2005, 2019):
    g = SF[SF.t0 == t0]
    rows.append(dict(t0=t0, **{c.replace('all_newapp_sh_', 'psi_'): psi(g[c].values, fin[c].values) for c in cols},
                     **{c.replace('all_newapp_sh_', 'mean_'): float(g[c].mean()) for c in cols}))
tr = SF[SF.t0.between(2005, 2018)]
rows.append(dict(t0='train_2005_2018', **{c.replace('all_newapp_sh_', 'psi_'): psi(tr[c].values, fin[c].values) for c in cols},
                 **{c.replace('all_newapp_sh_', 'mean_'): float(tr[c].mean()) for c in cols}))
rows.append(dict(t0='final_2021', **{c.replace('all_newapp_sh_', 'mean_'): float(fin[c].mean()) for c in cols}))
R = pd.DataFrame(rows); R.round(4).to_csv(os.path.join(CACHE, 's3b_psi_by_origin_scen.csv'), index=False)
print(R.round(3).to_string())

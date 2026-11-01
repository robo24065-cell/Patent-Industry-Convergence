import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r3lib, r3ext, numpy as np, pandas as pd
R = r3lib.R
d = R._read_frame('O18', 'ev.parquet', ['pid', 'L', 'ksic', 'next_annl', 'acd', 'age', 'atype', 'y'])
print(len(d), d.L.unique(), d.acd.dtype, flush=True)
for g in [['sep_self','sep_any','sep_a_share'], ['kp_n','kp_lxi'], ['pr_ppi_g1','pr_exp_g3'], ['np_pres','np_exit','np_gemp'], ['ap_late1y','ap_lapacc']]:
    t = time.time(); x = r3ext.add(d, g); print(g, f'{time.time()-t:.1f}s', flush=True)

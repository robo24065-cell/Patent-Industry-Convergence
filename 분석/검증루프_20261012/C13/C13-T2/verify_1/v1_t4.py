import sys, glob, os
import numpy as np, pandas as pd
base = sys.argv[1]; out = []
for f in sorted(glob.glob(os.path.join(base, 'cache', 't4', 'preds_b*.parquet'))):
    b = int(os.path.basename(f)[7:-8]); d = pd.read_parquet(f, columns=['y', 'p_champ', 'p_M0', 'p_strong'])
    y = d.y.values.astype(float); n = len(y); k = int(np.floor(0.01 * n + 0.5))
    r = dict(bundle=b, n=n, k=k)
    for c in ['p_champ', 'p_M0', 'p_strong']:
        p = d[c].values.astype(np.float64); thr = np.sort(p)[::-1][k - 1]
        ab = p > thr; tie = p == thr; need = k - ab.sum()
        r[c + '_prec_exp'] = (y[ab].sum() + need * y[tie].sum() / tie.sum()) / k
        r[c + '_ties'] = int(tie.sum())
        r[c + '_hit100'] = int(y[np.argsort(-p, kind='stable')[:100]].sum())
    out.append(r)
R = pd.DataFrame(out); pd.set_option('display.width', 250); print(R.to_string())
R.to_csv(os.path.join(base, 'verify_1', 'v1_t4_top1.csv'), index=False)

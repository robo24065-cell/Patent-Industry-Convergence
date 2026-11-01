import sys, time, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
import r3ext2, r3ext3
from sklearn.metrics import roc_auc_score
c = int(sys.argv[1]); r2.light_load()
te = pd.read_parquet(fc_r3(c)/'prep_te.parquet', columns=['firm', 'k', 'cut', 'y_u', 'prior_any', 't_lsum'])
tr = pd.read_parquet(fc_r3(c)/'prep_tr.parquet', columns=['firm', 'k', 'cut', 'y_reg', 't_lsum']); tr = tr[tr.cut == tr.cut.min()].reset_index(drop=True)
n0 = set(te.columns)
for w in sys.argv[2:]:
    t = time.time(); tr, te = {**r3ext2.EXT, **r3ext3.EXT}[w](tr, te, c); print(w, 'sec', round(time.time() - t), 'peak', round(peak_gb(), 2), flush=True)
te = te[~te.prior_any.values]
for x in [z for z in te.columns if z not in n0]:
    ok = te[x].notna()
    au = roc_auc_score(te.y_u[ok], te[x][ok]) if ok.sum() > 1000 else np.nan
    print(f'{x:12s} te nonnull {ok.mean():.3f} tr nonnull {tr[x].notna().mean():.3f} uniAUC {au:.3f}')

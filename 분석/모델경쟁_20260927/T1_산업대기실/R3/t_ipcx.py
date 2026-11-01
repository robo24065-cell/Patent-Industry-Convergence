import sys, time, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
import r3ext2
from sklearn.metrics import roc_auc_score
c = int(sys.argv[1]); r2.light_load()
te = pd.read_parquet(fc_r3(c)/'prep_te.parquet', columns=['firm', 'k', 'cut', 'y_u', 'prior_any', 'ipc_fit'] if False else ['firm', 'k', 'cut', 'y_u', 'prior_any'])
tr = pd.read_parquet(fc_r3(c)/'prep_tr.parquet', columns=['firm', 'k', 'cut', 'y_reg']); tr = tr[tr.cut == tr.cut.min()].reset_index(drop=True)
t = time.time(); [r3ext2.EXT[w](tr, te, c) for w in sys.argv[2:]]; print('sec', round(time.time() - t), 'peak', round(peak_gb(), 2))
te = te[~te.prior_any.values]
for x in [z for z in te.columns if z.startswith('ipcx')]:
    ok = te[x].notna(); print(x, 'te nonnull', round(ok.mean(), 3), 'tr nonnull', round(tr[x].notna().mean(), 3), 'uniAUC', round(roc_auc_score(te.y_u[ok], te[x][ok] * (1 if x == 'ipcx_fit' else -1)), 3))

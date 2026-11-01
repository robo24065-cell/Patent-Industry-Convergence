# B단계 외부 피처 점검(평가표 c=2012, 학습표 첫 원점 일부) — 연계율·결측률·소요시간
import sys, time, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
import r3ext2
c = int(sys.argv[1]); which = sys.argv[2:]
r2.light_load()
te = pd.read_parquet(fc_r3(c)/'prep_te.parquet', columns=['firm', 'k', 'cut', 't_lsum', 'y_u', 'prior_any'])
tr = pd.read_parquet(fc_r3(c)/'prep_tr.parquet', columns=['firm', 'k', 'cut', 't_lsum', 'y_reg'])
tr = tr[tr.cut == tr.cut.min()].reset_index(drop=True)
for w in which:
    t = time.time(); tr, te = r3ext2.EXT[w](tr, te, c)
    print(w, 'sec', round(time.time() - t), flush=True)
cols = [x for x in te.columns if x.split('_')[0] in ('oa', 'ros', 'ki')]
te = te[~te.prior_any.values]
for x in cols:
    v = te[x]; a = tr[x]
    ok = v.notna()
    from sklearn.metrics import roc_auc_score
    au = roc_auc_score(te.y_u[ok], v[ok]) if ok.sum() > 1000 and te.y_u[ok].nunique() > 1 else np.nan
    print(f'{x:10s} te_nonnull {ok.mean():.3f} mean {v.mean():.4f} tr_nonnull {a.notna().mean():.3f} tr_mean {a.mean():.4f} uniAUC {au:.3f}')
print('peak', round(peak_gb(), 2))

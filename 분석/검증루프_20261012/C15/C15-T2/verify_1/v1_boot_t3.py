# T3 묶음별 하한을 공통 난수(같은 군집 재표본)로 재계산 + T1 하한 규칙 확인
import sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import rankdata
B, OUT = sys.argv[1], sys.argv[2]
def auc(y, s):
    y = np.asarray(y).astype(bool); r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
sk = lambda a, a0: (a-a0)/(1-a0)
ds = [pd.read_parquet(f'{B}/cache/t3/preds_b{b}_final.parquet') for b in range(10)]
for d in ds[1:]: assert (d['cell'].values == ds[0]['cell'].values).all()
y = ds[0]['yb'].values; g = ds[0]['mg'].values; ug, inv = np.unique(g, return_inverse=True)
idx_by = [np.where(inv == i)[0] for i in range(len(ug))]
rng = np.random.default_rng(777); L = np.zeros((1000, 10)); L2 = np.zeros((1000, 10))
for r in range(1000):
    ii = np.concatenate([idx_by[i] for i in rng.integers(0, len(ug), len(ug))]); yy = y[ii]
    a0 = auc(yy, ds[0]['p_base'].values[ii])
    for b, d in enumerate(ds):
        a = auc(yy, d['p'].values[ii]); L[r, b] = sk(a, a0); L2[r, b] = sk(a, auc(yy, d['SB3'].values[ii]))
lo = np.percentile(L, 2.5, axis=0); lo2 = np.percentile(L2, 2.5, axis=0)
out = dict(T3_lo=lo.tolist(), T3_lo_sb3=lo2.tolist(), T3_lo_official_rank_from_bottom=int((lo < lo[0]).sum()+1), T3_lo_sb3_official_rank_from_bottom=int((lo2 < lo2[0]).sum()+1),
           T3_lo_range=float(lo.max()-lo.min()), T3_lo_sb3_range=float(lo2.max()-lo2.min()), T3_ci_width_se=float((np.percentile(L[:,0],97.5)-lo[0])/3.92))
# T1: 하한을 재표본별 max(xgb_y,xgb_y_u) 기준으로
sb = pd.read_csv(f'{B}/seed_bundles.csv'); s1 = sb[sb.track=='T1'].reset_index(drop=True)
t1 = []
for b in range(10):
    bm = np.vstack([np.load(f)['auc'] for f in sorted(glob.glob(f'{B}/cache/t1/boot/b{b}_m0_*.npz'))])
    # 후보 규칙: 각 묶음 점추정 M0 열 / 재표본별 max / 공식 b0 열(xgb_y_u)
    for name, a0 in [('max', bm[:,1:].max(1)), ('xgb_y', bm[:,1]), ('xgb_y_u', bm[:,2])]:
        s = (bm[:,0]-a0)/(1-a0); t1.append(dict(b=b, rule=name, lo=np.percentile(s,2.5), csv=s1.ci_lo[b], hi=np.percentile(s,97.5), csvhi=s1.ci_hi[b]))
t1 = pd.DataFrame(t1); t1['d'] = (t1.lo-t1.csv).abs(); out['T1_lo_rules'] = t1.groupby('rule')['d'].max().to_dict()
print(t1.to_string()); json.dump(out, open(f'{OUT}/v1_boot_result.json','w'), indent=1); print(json.dumps(out, indent=1))

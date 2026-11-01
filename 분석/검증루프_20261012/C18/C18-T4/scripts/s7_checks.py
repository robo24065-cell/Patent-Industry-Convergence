# C18-T4 s7: 재사용 자료 점검(data_check.txt용) — 중복·결측·KSIC 집중·유효 군집 수·시점. 출력 cache/s7_checks.json
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
R = {}
for fn in sorted(glob.glob(C + r'\fr_K*.parquet')):
    nm = os.path.basename(fn)[3:-8]
    F = pd.read_parquet(fn)
    K = nm.split('_')[0]
    key = {'K1': ['f', 's'], 'K2': ['f'], 'K3': ['f'], 'K4': ['c', 'st'] if 'st' in F else ['c']}[K]
    pos = F[F.y == 1]
    sh = pos.groupby('k').w.sum() / pos.w.sum()
    shn = F.groupby('k').w.sum() / F.w.sum()
    r = {'n': len(F), 'pos': int(F.y.sum()), 'dup_unit_key': int(F.duplicated(key).sum()), 'k_missing': int((F.k < 0).sum()),
         'nan_scores': int(F[[c for c in ['base', 'prov', 'free', 'hsc', 'scagg', 'off'] if c in F]].isna().sum().sum()),
         'n_k': int(F.k.nunique()), 'G_eff_pos': float(1 / (sh ** 2).sum()), 'G_eff_all': float(1 / (shn ** 2).sum()),
         'top1_k_pos_share': float(sh.max()), 'top10_k_pos_share': float(sh.sort_values(ascending=False).head(10).sum()),
         'k_base_rate_min': float((F.groupby('k').apply(lambda g: np.average(g.y, weights=g.w))).min()),
         'k_base_rate_max': float((F.groupby('k').apply(lambda g: np.average(g.y, weights=g.w))).max())}
    if K == 'K1':
        r['EP_share_units'] = float(F.cc.mean()); r['EP_share_pos'] = float(pos.cc.mean())
    R[nm] = r
    out(nm, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}))
json.dump(R, open(C + r'\s7_checks.json', 'w'), indent=1)

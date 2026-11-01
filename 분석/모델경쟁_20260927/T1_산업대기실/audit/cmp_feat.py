# final_run 재구축 피처 vs 2단계 실험(s2) 피처 일치 확인
import os, sys; os.environ['T1_STRICT'] = '1'; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2lib import *
spec = json.load(open(T1/'final_spec.json', encoding='utf-8')); cols = [c for c in spec['cols'] if c not in ('k_prior', 'k_prior_new', 'f_any')]
tr, te = load_s2()
for cut, ref in [(2009, te)] + [(c, tr[tr.cut == c].reset_index(drop=True)) for c in (2001, 2002, 2003, 2004)]:
    f = pd.read_parquet(T1/f'final_cache/dryrun/feat_c{cut}.parquet')
    assert (f.firm.values == ref.firm.values).all() and (f.k.values == ref.k.values).all() and (f.y.values == ref.y.values).all()
    bad = {c: float(np.nanmax(np.abs(f[c].values.astype(float) - ref[c].values.astype(float)))) for c in cols}
    nanm = {c: int((np.isnan(f[c].values.astype(float)) != np.isnan(ref[c].values.astype(float))).sum()) for c in cols}
    print(cut, 'rows', len(f), 'max|diff|>1e-5:', {c: v for c, v in bad.items() if v > 1e-5}, 'nan mismatch:', {c: v for c, v in nanm.items() if v})

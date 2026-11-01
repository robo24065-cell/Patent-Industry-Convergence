# T1 R3 외부데이터 연계율·단변량 진단(평가 원점 위험쌍 기준, 선행 공개출원 쌍 제외) → R3/ext_link_stats.csv
import sys; sys.path.insert(0, 'R3')
from r3 import *
import r3ext
r2.light_load(); rows = []
for c in (2008, 2012, 2015):
    te = pd.read_parquet(fc_r3(c)/'prep_te.parquet', columns=['firm', 'k', 'cut', 't_lsum', 'y_u', 'prior_any'])
    te = te[~te.prior_any.values].reset_index(drop=True); tr = te.iloc[:5].copy(); base = te.y_u.mean()
    nf = te.firm.nunique()
    for g, key in (('design', 'ds_lsum'), ('design', 'ds_n20'), ('um', 'um_lsum'), ('um', 'um_n20'), ('acq', 'acq_k'), ('acq', 'acq_n'), ('lic', 'lic_k'), ('tm', 'tm_n'),
                   ('sci', 'sci_c_g3'), ('price', 'pr_ppi_g3'), ('price', 'pr_exp_g3')):
        _, t2 = r3ext.EXT[g](tr.copy(), te[['firm', 'k', 'cut', 't_lsum', 'y_u']].copy(), c)
        v = t2[key].values; fin = np.isfinite(v); nz = fin & (v != 0)
        firm_cov = t2.loc[nz, 'firm'].nunique() / nf
        rows.append(dict(origin=c, source=g, feature=key, pairs=len(te), firms=nf, finite_pair_rate=round(fin.mean(), 4), nonzero_pair_rate=round(nz.mean(), 4),
                         firm_link_rate=round(firm_cov, 4), ksic_covered=int(t2.loc[fin, 'k'].nunique()),
                         entry_rate_nonzero=round(t2.y_u.values[nz].mean(), 4) if nz.any() else None, entry_rate_all=round(base, 4),
                         lift=round(t2.y_u.values[nz].mean() / base, 2) if nz.any() else None,
                         uni_auc=round(roc_auc_score(t2.y_u.values[fin], v[fin]), 4)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('R3/ext_link_stats.csv', index=False, encoding='utf-8-sig')

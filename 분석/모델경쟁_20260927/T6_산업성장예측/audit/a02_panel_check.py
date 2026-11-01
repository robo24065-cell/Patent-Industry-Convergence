# 감사 2: final_run.build_panel(2019,2019)이 1단계 panel_base.parquet를 재현하는지 + 시점 점검(g1(T)=y(T-1))
import sys, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H); os.chdir(H)
import pandas as pd, numpy as np, time
import final_run as fr
t = time.time()
F = fr.build_panel(2019, 2019, extra=True)
print('build sec', round(time.time() - t, 1), F.shape, 'max year', F.year.max())
F.to_parquet('cache/panel_v2_2019.parquet', index=False)
O = pd.read_parquet('cache/panel_base.parquet')
m = O.merge(F, on=['ksic', 'year'], suffixes=('_o', '_n'))
print('rows old', len(O), 'new', len(F), 'merged', len(m))
bad = []
for c in O.columns:
    if c in ('ksic', 'year') or c + '_n' not in m: continue
    a, b = m[c + '_o'], m[c + '_n']
    if a.dtype == object:
        eq = (a.fillna('') == b.fillna('')).all()
    else:
        eq = np.allclose(a.astype(float).fillna(-999), b.astype(float).fillna(-999), atol=1e-9)
    if not eq: bad.append(c)
print('불일치 컬럼:', bad)
# 시점 점검
s = F.sort_values(['ksic', 'year'])
lagy = s.groupby('ksic').y.shift(1)
chk = s[s.year.between(2013, 2019)]
print('g1==y(T-1):', np.allclose(chk.g1.fillna(-9), lagy[chk.index].fillna(-9)))
print('new extra coverage (SECT, 2012-2019):')
S = F[F.k2.isin(['15','20','21','23','24','25','26','27','28','29','30','31','33']) & F.year.between(2012, 2019)]
print(S[['cu_q4x', 'cu_gx', 'exp_q4x', 'dom_q4x', 'expsh_gx', 'fin_sales', 'fin_inv', 'fin_inv_gap']].notna().mean().round(3).to_dict())

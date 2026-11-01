# 새 학습 원점의 모든 피처 캐시 생성: python build_cut.py 2001
import sys
from data8 import *
from feats import agg_feats, industry_mom, partner_feats
cut = int(sys.argv[1]); cap = CAP_TR if len(sys.argv) < 3 else None; t0 = time.time()
capn = 'None' if cap is None else cap
P = train_clf(CAP_TR, 'c2009_base'); panel(cut, cap, 'c2009_base', P)
P51, PO = train_clf51(CAP_TR, 'c2009_os51')
p = PC/f'panel_c{cut}_{capn}_c2009_os51_.parquet'
if not p.exists():
    d, c = build_panel(cut, 5, cap=cap); d = add_dart(d, c); d = agg_feats(d, c, P51); d = industry_mom(d, c)
    op = c['op']; rows = c['fi'][op.applicant_cd].values
    s50 = np.bincount(rows, weights=1 - PO[op.row.values], minlength=len(c['firms'])) / np.maximum(np.bincount(rows, minlength=len(c['firms'])), 1)
    d['in50'] = s50[d.fidx.values]; d['cut'] = cut; d.to_parquet(p, index=False)
pp = PC/f'part_c{cut}_{capn}.parquet'
if not pp.exists():
    d, c = build_panel(cut, 5, cap=cap); partner_feats(d[['fidx', 'k']].copy(), c)[['p_n', 'p_k', 'p_share']].to_parquet(pp, index=False)
sib_panel(cut, capn); ins_panel(cut, capn); tim_panel(cut, capn)
print('built', cut, round(time.time() - t0))

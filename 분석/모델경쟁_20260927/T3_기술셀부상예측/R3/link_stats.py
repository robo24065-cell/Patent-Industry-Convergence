"""외부자료 연계율(적격 셀 기준)·원점별 단변량 AUC(개발 원점만, 기술 통계). 결과: ext3_link_eligible_{var}.csv"""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument('--var', default='pub'); a = ap.parse_args(); os.environ['T3R3_VAR'] = a.var
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, r3core as C
from sklearn.metrics import roc_auc_score
D = C.get_D()
fe = os.path.join(C.L.CACHE, 'early_t0.parquet')
if a.var == 'pub' and os.path.exists(fe): D = D.merge(pd.read_parquet(fe), on=['cell', 't0'], how='left')
dev = D[D.t0.isin(C.DEV_T0)]
cols = ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2', 'oa_all_g', 'oa_fx_g', 'oa_kr_g', 'oa_comp_g', 'oa_comp_sh', 'oa_kr_sh', 'pr_exp_g3', 'pr_imp_g3', 'pr_ppi_g3', 'pr_ppi_g1', 'pr_tot',
        'us_rship_g3', 'us_emp_g3', 'us_tfp_g3', 'ip5_kr_sh', 'ip5_kr_sh_chg', 'ip5_w_g', 'kp_n', 'kp_lxi', 'kp_lxi_rel', 'kr_early_n', 'kr_early_sh', 'kr_appc_mom', 'fx_appc_mom', 'fx_early_sh']
rows = []
for c in cols:
    if c not in dev: continue
    au = []
    for t, g in dev.groupby('t0'):
        m = g[c].notna()
        if m.sum() > 50 and g.yb[m].nunique() == 2: au.append(roc_auc_score(g.yb[m], g[c][m]))
    rows.append(dict(feature=c, eligible_cell_link_rate=round(float(dev[c].notna().mean()), 3), ksic_linked=int(dev.groupby('ksic')[c].apply(lambda s: s.notna().any()).sum()),
                     uni_auc_mean=round(float(np.mean(au)), 4) if au else np.nan, uni_auc_min=round(float(np.min(au)), 4) if au else np.nan, uni_auc_max=round(float(np.max(au)), 4) if au else np.nan))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, f'ext3_link_eligible_{a.var}.csv'), index=False, encoding='utf-8-sig'); print(R.to_string())
print('eligible cells dev', len(dev), 'by t0', dev.groupby('t0').size().to_dict())

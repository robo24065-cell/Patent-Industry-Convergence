# 2단 계층 모형: (1) 권리자×L 수준 '매각률' 모형 (2) 특허 수준 v12 의 권리자 내 상대순위.
# p_final = owner_hat × p_patent / mean_owner(p_patent)
# python owner_model.py <version> <weight: 1|sqrt|port> <featset: A|B> <patent_pred_version>
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import xgboost as xgb
import run
from common import *
from evalkit import *
ver, wmode, fset, pv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
AGG_MEAN = ['age', 'lclaims', 'inv_is_app', 'prior_MAIN', 'cur_is_orig', 'paid_thru', 'prepaid', 'for_prio', 'pct', 'tech_exit', 'k_ipc_app3',
            'own_ipc_share', 'lic_any', 'n_inv', 'exam_yrs', 'ksic_rate3', 'multi', 'amt_per_yr', 'rgst_year']
OWN_LEVEL = ['own_port', 'own_lapse3', 'own_lapse1', 'own_lapse_rate', 'own_lapse_trend', 'own_corp3', 'own_in3', 'own_in1', 'own_app3', 'own_app_prev',
             'own_app_mom', 'own_years', 'own_n_ksic', 'k_app3', 'k_app_prev', 'k_mom', 'k_dormant', 'cur_type', 'cur_n']
SALE = ['own_out3', 'own_out3_days', 'own_out_all', 'own_out_rate', 'own_out1', 'own_out6m']


def owner_table(d, with_y=True):
    g = d.groupby(['L', 'cur'])
    o = g[AGG_MEAN].mean().add_prefix('m_')
    o = o.join(g[OWN_LEVEL + SALE].first())
    o['ksic_mode'] = g.ksic_c.agg(lambda s: s.value_counts().index[0])
    o['n'] = g.size()
    if with_y: o['rate'] = g.ytr.mean()
    return o.reset_index()


tr = run.load(range(2006, TRAIN_MAX_L + 1)); va = run.load(VAL_L); tr['ytr'] = tr.y_MAIN
ot = owner_table(tr); ov = owner_table(va, with_y=False)
F = ['m_' + c for c in AGG_MEAN] + OWN_LEVEL + ['ksic_mode'] + (SALE if fset == 'B' else [])
w = {'1': np.ones(len(ot)), 'sqrt': np.sqrt(ot.n), 'port': ot.n.astype(float)}[wmode]
m = xgb.XGBRegressor(objective='reg:logistic', n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                     min_child_weight=5, reg_lambda=5.0, nthread=2, tree_method='hist', random_state=0)
m.fit(ot[F], ot.rate, sample_weight=w)
ov['owner_hat'] = m.predict(ov[F])
print('owners train', len(ot), 'val', len(ov))
pp = pd.read_parquet(f'cache/pred_{pv}.parquet')
d = va[['rgst_no', 'L', 'cur', 'y_MAIN']].merge(pp, on=['rgst_no', 'L']).merge(ov[['L', 'cur', 'owner_hat']], on=['L', 'cur'])
d['rel'] = d.p / d.groupby(['L', 'cur']).p.transform('mean')
d['pf'] = d.owner_hat * d.rel
base = json.load(open('cache/v0_metrics.json'))['y_MAIN']
for nm, col in [('owner_hat only', 'owner_hat'), ('final', 'pf')]:
    mt = metrics(d.y_MAIN, d[col]); print(nm, {k: round(x, 4) for k, x in mt.items()})
mt = metrics(d.y_MAIN, d.pf); ci = cluster_boot(d.y_MAIN.values, d.pf.values, d.cur.values)
print('CI', ci[0].round(4), ci[1].round(4))
for L in VAL_L:
    k = (d.L == L).values; print('  L', L, {kk: round(x, 4) for kk, x in metrics(d.y_MAIN[k], d.pf[k]).items()})
imp = pd.Series(m.get_booster().get_score(importance_type='gain')).sort_values(ascending=False); print('owner gain top', imp.head(12).round(2).to_dict())
d[['rgst_no', 'L', 'pf']].rename(columns={'pf': 'p'}).to_parquet(f'cache/pred_{ver}.parquet', index=False)
json.dump(dict(auc=mt['auc'], prauc=mt['prauc'], p1=mt['p1'], ci_auc=list(ci[0]), ci_pr=list(ci[1])), open(f'cache/metrics_{ver}.json', 'w'))
print('RESULT', ver, round(mt['auc'], 4), round(mt['prauc'], 4), round(mt['p1'], 4), ci[0].round(3))

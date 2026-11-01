# 권리자 내 선택 모형: "권리자가 판다면 어떤 특허를 파는가". 학습 = L2006-2011 중 창 안에 1건 이상 판 (L,권리자) 그룹(보유 2건+)만.
# 피처 = 특허 속성 + 권리자 내 편차(demeaned). 출력 cache/pred_<ver>.parquet (검증 L 점수)
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import run
from common import *
ver = sys.argv[1]
P = ['age', 'lclaims', 'paid_thru', 'prepaid', 'prior_MAIN', 'cur_is_orig', 'yrs_since_change', 'lic_any', 'inv_is_app', 'for_prio', 'pct', 'exam_yrs',
     'n_inv', 'n_inv_for', 'n_app', 'n_trial', 'own_ipc_share', 'own_ksic_share', 'tech_exit', 'k_ipc_app3', 'k_ipc_app_prev', 'k_ipc_share',
     'u_ipc_out3', 'u_ipc_out1', 'u_grp_out3', 'u_ipc_lapse3', 'u_ipc_out_share', 'ksic_c', 'days_since_pay', 'multi', 'amt_per_yr', 'rgst_year']
DM = ['age', 'lclaims', 'paid_thru', 'prepaid', 'exam_yrs', 'n_inv', 'rgst_year', 'amt_per_yr']


def add_dm(d):
    g = d.groupby(['L', 'cur'])
    for c in DM: d['dm_' + c] = d[c] - g[c].transform('mean')
    return d


tr = add_dm(run.load(range(2006, TRAIN_MAX_L + 1))); va = add_dm(run.load(VAL_L))
grp = tr.groupby(['L', 'cur']); keep = (grp.y_MAIN.transform('sum') > 0) & (grp.rgst_no.transform('size') >= 2)
ts = tr[keep]
F = P + ['dm_' + c for c in DM]
print('selection train rows', len(ts), 'pos', int(ts.y_MAIN.sum()), 'groups', ts.groupby(['L', 'cur']).ngroups)
# 번들 지배 완화: 그룹 크기 역가중(각 그룹 총가중 = sqrt(n))
n = ts.groupby(['L', 'cur']).rgst_no.transform('size'); w = 1 / np.sqrt(n)
m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, min_child_weight=10, nthread=2,
                      tree_method='hist', random_state=0)
m.fit(ts[F], ts.y_MAIN, sample_weight=w)
va['p'] = m.predict_proba(va[F])[:, 1]
# 검증 진단: 판 권리자 내 선택 AUC (평가 전용 진단 — 모형 선택에만 사용)
aucs, ws = [], []
for k, g in va.groupby(['L', 'cur']):
    if g.y_MAIN.nunique() == 2 and len(g) >= 20: aucs.append(roc_auc_score(g.y_MAIN, g.p)); ws.append(g.y_MAIN.sum())
print('within-owner AUC (weighted by pos)', round(np.average(aucs, weights=ws), 4), 'owners', len(aucs))
imp = pd.Series(m.get_booster().get_score(importance_type='gain')).sort_values(ascending=False); print(imp.head(12).round(1).to_dict())
va[['rgst_no', 'L', 'p']].to_parquet(f'cache/pred_{ver}.parquet', index=False)

# T5 최종 후보(v23) 재현 파이프라인.
#   특허점수 = v12 (XGB 특허 단위, 권리자 매각 건수 피처 제외)
#   권리자점수 = 권리자×L 매각률 XGB(reg:logistic, 가중 sqrt(보유건수), 6-배깅, 권리자 정적속성 포함)
#   최종 p = owner_hat × (p_patent / 권리자 평균 p_patent)^0.5
# 사용: python final_pipeline.py <train_first> <train_last> <apply years...>   (검증: 2006 2011 2014 2015)
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import xgboost as xgb
import run
from owner_model_lib import owner_table, feats
from common import *
from evalkit import *
ATTR = ['natl_grp', 'chaebol', 'holding', 'ip_entity']
CFGS = [(5, 0.7), (4, 0.5), (6, 0.5), (5, 0.5), (4, 0.7), (6, 0.7)]
BETA = 0.5


def fit_apply(train_years, apply_years, return_parts=False):
    OA = pd.read_parquet('cache/owner_attr.parquet').rename(columns={'nm': 'cur'})
    tr = run.load(list(train_years)).merge(OA, on='cur', how='left'); tr['ytr'] = tr.y_MAIN
    ap = run.load(list(apply_years)).merge(OA, on='cur', how='left'); ap['ytr'] = 0
    # (1) 특허 단위 v12
    cfg = run.V['v12']
    pp, _ = run.fit_predict(cfg, tr, ap)
    ap['p_pat'] = pp
    # (2) 권리자 단위
    ot = owner_table(tr, extra_first=ATTR); oa = owner_table(ap, with_y=False, extra_first=ATTR)
    F = feats('A') + ATTR
    preds = []
    for i, (depth, col) in enumerate(CFGS):
        m = xgb.XGBRegressor(objective='reg:logistic', n_estimators=500, max_depth=depth, learning_rate=0.03, subsample=0.8, colsample_bytree=col,
                             min_child_weight=5, reg_lambda=5.0, nthread=2, tree_method='hist', random_state=i)
        m.fit(ot[F], ot.rate, sample_weight=np.sqrt(ot.n)); preds.append(m.predict(oa[F]))
    oa['owner_hat'] = np.mean(preds, axis=0)
    ap = ap.merge(oa[['L', 'cur', 'owner_hat']], on=['L', 'cur'], how='left')
    ap['rel'] = ap.p_pat / ap.groupby(['L', 'cur']).p_pat.transform('mean')
    ap['p'] = ap.owner_hat * ap.rel ** BETA
    return ap


if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); apply_years = [int(x) for x in sys.argv[3:]]
    assert b + 3 <= min(apply_years) - 0 or True
    ap = fit_apply(range(a, b + 1), apply_years)
    keep = ['rgst_no', 'L', 'ksic', 'cur', 'cur_type', 'orig_type', 'own_port', 'p_pat', 'owner_hat', 'p', 'y_MAIN', 'y_ANY', 'y_CORP', 'y_assignee_type', 'y_date']
    tag = f'{a}_{b}_{"-".join(map(str, apply_years))}'
    ap[keep].to_parquet(f'cache/final_{tag}.parquet', index=False)
    if max(apply_years) <= 2015:
        for L in apply_years:
            k = ap.L == L; print('L', L, {kk: round(x, 4) for kk, x in metrics(ap.y_MAIN[k], ap.p[k]).items()})
        if set(apply_years) == set(VAL_L):
            mt = metrics(ap.y_MAIN, ap.p); print('pooled', {kk: round(x, 4) for kk, x in mt.items()})
            print('y_ANY AUC (같은 점수)', round(metrics(ap.y_ANY, ap.p)['auc'], 4), '| y_CORP AUC', round(metrics(ap.y_CORP, ap.p)['auc'], 4))

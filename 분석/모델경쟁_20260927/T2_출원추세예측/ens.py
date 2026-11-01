# 저장된 검증 예측으로 앙상블 평가 (사전 지정 조합, 동일가중)
import sys, itertools
import numpy as np, pandas as pd
from common import *

nb = naive_ref()
NB = summarize(nb)['MASE']


def load(n):
    r = pd.read_parquet(os.path.join(T, 'preds', f'{n}.parquet'))
    assert (r[['ksic', 'cc', 'origin', 'h']].values == nb[['ksic', 'cc', 'origin', 'h']].values).all()
    return r


def combo(names, how='mean'):
    F = np.stack([load(n).f.values for n in names])
    if how == 'mean':
        f = F.mean(0)
    elif how == 'median':
        f = np.median(F, 0)
    elif how == 'geo':
        f = np.expm1(np.log1p(F).mean(0))
    r = nb.copy(); r['f'] = f; r['ae'] = (r.y - r.f).abs(); r['ase'] = r.ae / r.scale
    return r


if __name__ == '__main__':
    sets = {
        'E_A': ['s_ridge_main_only', 'topdown_T8_1.0', 'flow_lag1_rk1'],
        'E_B': ['logdrift_k8_p0.8', 'theta_log', 's_ridge_main_only', 'topdown_T8_1.0', 'flow_lag1_rk1'],
        'E_C': ['logdrift_k8_p0.8', 'theta_log', 's_ridge_main_only', 'topdown_T8_1.0', 'flow_lag1_rk1', 'g_all_ridge', 'g_xc_ctot_ktot_xgb', 'arima_log_2005', 'ets_damped'],
    }
    for nm, s in sets.items():
        for how in ['mean', 'median', 'geo']:
            r = combo(s, how)
            report(r, f'{nm}_{how}', seg=(how == 'mean'))
            save_pred(r, f'{nm}_{how}')

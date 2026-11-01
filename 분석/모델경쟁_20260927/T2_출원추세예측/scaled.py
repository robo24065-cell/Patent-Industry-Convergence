# MASE 공간 직접 학습: 타깃 = (y_{t+h} - y_t) / scale_t, scale_t = 기준연도 t 까지 1-step 평균절대차
# 손실을 평가지표(MASE)와 일치시킴
import sys, json, warnings
import numpy as np, pandas as pd
from common import *
from flow import FY, SRC, CCI

warnings.filterwarnings('ignore')
MAIN = load_panel()
CTOT = MAIN.groupby(level='cc').sum().values


def feats(n, c, fs):
    sc = np.mean(np.abs(np.diff(n[:, :c + 1], axis=1)), axis=1); sc = np.where(sc <= 0, 1, sc)
    L = np.log1p(n)
    F = {}
    for l in range(1, 6):
        F[f'd_l{l}'] = (n[:, c - l + 1] - n[:, c - l]) / sc
    F['d3'] = (n[:, c] - n[:, c - 3]) / 3 / sc
    F['lvl_sc'] = np.log(n[:, c] / sc + 1)
    for l in range(1, 4):
        F[f'g_l{l}'] = L[:, c - l + 1] - L[:, c - l]
    F['g3'] = (L[:, c] - L[:, c - 3]) / 3
    F['g5'] = (L[:, c] - L[:, c - 5]) / 5
    # 로그성장 외삽의 MASE 공간 환산(비선형 핵심 피처)
    F['ld_proj1'] = n[:, c] * (np.exp(F['g5']) - 1) / sc
    F['ld_proj3'] = n[:, c] * (np.exp(F['g3']) - 1) / sc
    Lt = np.log1p(CTOT)
    F['c_g1'] = (Lt[:, c] - Lt[:, c - 1])[CCI]
    F['c_g3'] = ((Lt[:, c] - Lt[:, c - 3]) / 3)[CCI]
    F['c_proj'] = n[:, c] * (np.exp(F['c_g3']) - 1) / sc
    for j, cj in enumerate(CCS):
        F[f'is_{cj}'] = (CCI == j).astype(float)
    if 'lead' in fs:
        nn = np.maximum(n[:, c], 1); lead = np.zeros(len(n))
        for Y in CCS:
            lead += FY[Y][:, c] / nn * (np.log1p(SRC[Y][:, c]) - np.log1p(SRC[Y][:, c - 1]))
        F['lead_g1'] = lead
        F['lead_proj'] = n[:, c] * lead / sc
    return pd.DataFrame(F), sc


def sc_fn(hist, o, kind='huber', fs=('lead',), tmin=2005, alpha=1.0, clip=None):
    n = hist
    out = np.zeros((n.shape[0], H))
    for h in range(1, H + 1):
        X, y = [], []
        for t in range(tmin, o - h + 1):
            c = col(t)
            f, sc = feats(n, c, fs)
            X.append(f); y.append((n[:, c + h] - n[:, c]) / sc)
        X = pd.concat(X, ignore_index=True); y = np.concatenate(y)
        if clip is not None:
            lo, hi = np.quantile(y, [clip, 1 - clip]); y = np.clip(y, lo, hi)
        Xo, sco = feats(n, col(o), fs)
        Xo = Xo[X.columns]
        from sklearn.preprocessing import StandardScaler
        if kind == 'huber':
            from sklearn.linear_model import HuberRegressor
            ss = StandardScaler().fit(X.values)
            m = HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000).fit(ss.transform(X.values), y)
            d = m.predict(ss.transform(Xo.values))
        elif kind == 'lad':
            from sklearn.linear_model import QuantileRegressor
            ss = StandardScaler().fit(X.values)
            m = QuantileRegressor(quantile=0.5, alpha=alpha * 1e-3, solver='highs').fit(ss.transform(X.values), y)
            d = m.predict(ss.transform(Xo.values))
        else:
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=10, reg_lambda=5.0, n_jobs=2, tree_method='hist',
                                 objective='reg:absoluteerror').fit(X.values, y)
            d = m.predict(Xo.values)
        out[:, h - 1] = n[:, -1] + d * sco
    return np.maximum(out, 0)


if __name__ == '__main__':
    runs = [('sc_huber', {}), ('sc_lad', {'kind': 'lad'}), ('sc_xgb', {'kind': 'xgb'}), ('sc_huber_nolead', {'fs': ()})]
    if len(sys.argv) > 1:
        runs = [(f'sc_custom{i}', json.loads(a)) for i, a in enumerate(sys.argv[1:])]
    for nm, kw in runs:
        r = evaluate(sc_fn, **kw)
        report(r, nm, seg=True)
        save_pred(r, nm)

# 통계 시계열 모형군 (시계열별 개별 적합). 사용: python stat_models.py <name> [<name> ...]
import sys, warnings
import numpy as np
from common import *

warnings.filterwarnings('ignore')


def drift_fn(hist, o, start=Y0):
    h0 = hist[:, col(start):]
    slope = (h0[:, -1] - h0[:, 0]) / (h0.shape[1] - 1)
    return hist[:, -1:] + slope[:, None] * np.arange(1, H + 1)[None, :]


def ets_fn(hist, o, log=False, damped=True, start=Y0):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    out = np.zeros((hist.shape[0], H))
    for i in range(hist.shape[0]):
        y = hist[i, col(start):]
        yy = np.log1p(y) if log else y
        try:
            m = ExponentialSmoothing(yy, trend='add', damped_trend=damped, initialization_method='estimated').fit()
            f = m.forecast(H)
        except Exception:
            f = np.repeat(yy[-1], H)
        out[i] = np.expm1(f) if log else f
    return np.maximum(out, 0)


def theta_fn(hist, o, start=Y0, log=True):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    out = np.zeros((hist.shape[0], H))
    for i in range(hist.shape[0]):
        y = hist[i, col(start):]
        yy = np.log1p(y) if log else y
        try:
            f = ThetaModel(yy, period=1, deseasonalize=False).fit().forecast(H)
            f = np.asarray(f)
        except Exception:
            f = np.repeat(yy[-1], H)
        out[i] = np.expm1(f) if log else f
    return np.maximum(out, 0)


def arima_fn(hist, o, start=Y0):
    from statsmodels.tsa.arima.model import ARIMA
    out = np.zeros((hist.shape[0], H))
    for i in range(hist.shape[0]):
        yy = np.log1p(hist[i, col(start):])
        best = None
        for order, trend in [((0, 1, 0), 't'), ((1, 1, 0), 't'), ((0, 1, 1), 't'), ((1, 1, 0), 'n'), ((0, 1, 1), 'n'), ((1, 1, 1), 'n')]:
            try:
                m = ARIMA(yy, order=order, trend=trend).fit()
                if best is None or m.aicc < best.aicc:
                    best = m
            except Exception:
                pass
        f = best.forecast(H) if best is not None else np.repeat(yy[-1], H)
        out[i] = np.expm1(np.asarray(f))
    return np.maximum(out, 0)


def logdrift_fn(hist, o, k=5, phi=1.0):
    """최근 k년 평균 로그성장률을 phi로 감쇠해 외삽"""
    L = np.log1p(hist)
    g = (L[:, -1] - L[:, -1 - k]) / k
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.expm1(L[:, -1:] + g[:, None] * steps[None, :])


def lindrift_fn(hist, o, k=5, phi=1.0):
    """최근 k년 평균 절대증가분을 phi로 감쇠해 외삽"""
    g = (hist[:, -1] - hist[:, -1 - k]) / k
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.maximum(hist[:, -1:] + g[:, None] * steps[None, :], 0)


MODELS = {
    'naive': (naive_fn, {}),
    'drift': (drift_fn, {}),
    'ets_damped': (ets_fn, {}),
    'ets_damped_log': (ets_fn, {'log': True}),
    'ets_damped_2005': (ets_fn, {'start': 2005}),
    'theta_log': (theta_fn, {}),
    'theta_log_2005': (theta_fn, {'start': 2005}),
    'arima_log': (arima_fn, {}),
    'arima_log_2005': (arima_fn, {'start': 2005}),
}
for k in [3, 5, 8]:
    for phi in [1.0, 0.8, 0.6, 0.4]:
        MODELS[f'logdrift_k{k}_p{phi}'] = (logdrift_fn, {'k': k, 'phi': phi})
        MODELS[f'lindrift_k{k}_p{phi}'] = (lindrift_fn, {'k': k, 'phi': phi})

def logols_fn(hist, o, k=8, phi=0.8, anchor='last', robust=False):
    """최근 k년 로그값에 OLS(또는 Theil-Sen) 기울기 → 감쇠외삽. anchor='fit'이면 적합선 끝점에서 출발"""
    L = np.log1p(hist[:, -k:])
    x = np.arange(k) - (k - 1) / 2
    if robust:
        from scipy.stats import theilslopes
        g = np.array([theilslopes(L[i], np.arange(k))[0] for i in range(L.shape[0])])
    else:
        g = (L * x[None, :]).sum(1) / (x ** 2).sum()
    start = L[:, -1] if anchor == 'last' else L.mean(1) + g * (k - 1) / 2
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.expm1(start[:, None] + g[:, None] * steps[None, :])


for k in [5, 8, 10, 12]:
    for phi in [1.0, 0.8, 0.6]:
        MODELS[f'logols_k{k}_p{phi}'] = (logols_fn, {'k': k, 'phi': phi})
        MODELS[f'logols_fit_k{k}_p{phi}'] = (logols_fn, {'k': k, 'phi': phi, 'anchor': 'fit'})
    MODELS[f'logts_k{k}_p0.8'] = (logols_fn, {'k': k, 'phi': 0.8, 'robust': True})


if __name__ == '__main__':
    names = sys.argv[1:]
    if names == ['grid2']:
        names = [n for n in MODELS if n.startswith(('logols', 'logts'))]
    if names == ['grid']:
        names = [n for n in MODELS if n.startswith(('logdrift', 'lindrift'))]
    for n in names:
        fn, kw = MODELS[n]
        r = evaluate(fn, **kw)
        report(r, n, seg=len(names) <= 4)
        save_pred(r, n)

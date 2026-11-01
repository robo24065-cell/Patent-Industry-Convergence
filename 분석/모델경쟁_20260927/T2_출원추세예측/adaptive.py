# 적응형 결합: 각 원점 o 에서 "목표연도 ≤ o" 인 내부 백테스트(원점 2008..o-1)로 후보모형의 국가×h별 성과를 측정해
# 가중/선택. 원점 o 이전 정보만 쓰므로 홀드아웃(원점 2018)에도 같은 규칙으로 적용 가능.
import sys, json, warnings, pickle
import numpy as np, pandas as pd
from common import *

warnings.filterwarnings('ignore')
MAIN = load_panel()
Yv = MAIN.values.astype(float)
CCI = np.array([CCS.index(c) for c in MAIN.index.get_level_values('cc')])
BT_ORIGINS = list(range(2008, 2016))


def candidates():
    from stat_models import logdrift_fn, lindrift_fn, theta_fn
    from hier import topdown_fn
    from flow import flow_fn
    from global_sub import sub_fn
    return {
        'naive': (naive_fn, {}),
        'ld8_.8': (logdrift_fn, {'k': 8, 'phi': 0.8}),
        'ld5_.8': (logdrift_fn, {'k': 5, 'phi': 0.8}),
        'ld3_.6': (logdrift_fn, {'k': 3, 'phi': 0.6}),
        'ld8_.5': (logdrift_fn, {'k': 8, 'phi': 0.5}),
        'lin5': (lindrift_fn, {'k': 5, 'phi': 1.0}),
        'td': (topdown_fn, {'kt': 8, 'pt': 1.0}),
        'flow': (flow_fn, {'rk': 1}),
        'ridge': (sub_fn, {'kind': 'ridge', 'minavg': 1e12}),
        'theta': (theta_fn, {}),
    }


def backtest_cache():
    p = os.path.join(CACHE, 'bt_forecasts.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    out = {}
    for nm, (fn, kw) in candidates().items():
        for o in BT_ORIGINS:
            assert o + H <= 2018
            out[(nm, o)] = np.asarray(fn(Yv[:, :col(o) + 1].copy(), o, **kw), float)
        print('bt', nm, flush=True)
    pickle.dump(out, open(p, 'wb'))
    return out


BT = None


def scaled_err(o, F):
    hist = Yv[:, :col(o) + 1]
    sc = np.mean(np.abs(np.diff(hist, axis=1)), axis=1); sc = np.where(sc <= 0, 1, sc)
    E = np.full((Yv.shape[0], H), np.nan)
    for h in range(1, H + 1):
        E[:, h - 1] = np.abs(Yv[:, col(o + h)] - F[:, h - 1]) / sc
    return E


def adaptive_fn(hist, o, names=None, mode='inv', topn=3, min_inner=2, pooled_h=False, by='cc', power=1.0, window=None):
    names = names or [n for n in candidates()]
    out = np.zeros((hist.shape[0], H))
    inner = [oi for oi in BT_ORIGINS if oi < o and (window is None or oi >= o - window)]
    # err[n][h] : 250 x (#inner with target<=o)
    for h in range(1, H + 1):
        hs = range(1, H + 1) if pooled_h else [h]
        S = {}
        for n in names:
            vals = []
            for oi in inner:
                E = scaled_err(oi, BT[(n, oi)])
                for hh in hs:
                    if oi + hh <= o:  # 원점 o 시점에 관측 완료된 목표만
                        vals.append(E[:, hh - 1])
            S[n] = np.stack(vals, 1) if len(vals) >= min_inner else None
        valid = [n for n in names if S[n] is not None]
        if not valid:
            valid = ['naive']
        groups = [np.where(CCI == j)[0] for j in range(5)] if by == 'cc' else [np.arange(len(CCI))]
        for g in groups:
            sc = np.array([np.nanmean(S[n][g]) if S.get(n) is not None else np.inf for n in valid])
            if mode == 'inv':
                w = (1 / sc) ** power
            elif mode == 'best':
                w = (sc == sc.min()).astype(float)
            elif mode == 'top':
                w = np.zeros(len(valid)); w[np.argsort(sc)[:topn]] = 1
            w = w / w.sum()
            Fo = np.stack([BT[(n, o)][g, h - 1] for n in valid])
            out[g, h - 1] = (w[:, None] * Fo).sum(0)
    return out


if __name__ == '__main__':
    BT = backtest_cache()
    runs = [('ad_inv_cc', {}), ('ad_best_cc', {'mode': 'best'}), ('ad_top3_cc', {'mode': 'top'}),
            ('ad_inv2_cc', {'power': 2}), ('ad_top3_all', {'mode': 'top', 'by': 'all'}),
            ('ad_top3_cc_pooledh', {'mode': 'top', 'pooled_h': True}), ('ad_inv_all', {'by': 'all'})]
    if len(sys.argv) > 1:
        runs = [(f'custom{i}', json.loads(a)) for i, a in enumerate(sys.argv[1:])]
    for nm, kw in runs:
        r = evaluate(adaptive_fn, **kw)
        report(r, nm, seg=True)
        save_pred(r, nm)

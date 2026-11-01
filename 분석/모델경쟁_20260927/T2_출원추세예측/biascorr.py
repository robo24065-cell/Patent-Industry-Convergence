# 편향 보정: 원점 o 에서 목표≤o 내부 백테스트(원점 2008..o-1)의 국가×h별 평균 로그편향을 계산해 앙상블 예측을 곱셈 보정
import sys, json, pickle, warnings
import numpy as np, pandas as pd
from common import *
import adaptive as A

warnings.filterwarnings('ignore')
A.BT = A.backtest_cache()
Yv = A.Yv; CCI = A.CCI


def ens_bt(o, members):
    return np.mean([A.BT[(m, o)] for m in members], 0)


def bc_fn(hist, o, members=('ridge', 'td', 'flow'), shrink=0.5, by='cc', window=None, kind='log'):
    F = ens_bt(o, members)
    out = F.copy()
    for h in range(1, H + 1):
        lb = []
        for oi in A.BT_ORIGINS:
            if oi + h <= o and (window is None or oi >= o - window):
                Fi = ens_bt(oi, members)[:, h - 1]
                lb.append(np.log1p(Fi) - np.log1p(Yv[:, col(oi + h)]))
        if not lb:
            continue
        lb = np.stack(lb, 1)  # 250 x m
        groups = [np.where(CCI == j)[0] for j in range(5)] if by == 'cc' else [np.arange(len(CCI))]
        for g in groups:
            b = np.median(lb[g]) if kind == 'median' else lb[g].mean()
            out[g, h - 1] = np.expm1(np.log1p(F[g, h - 1]) - shrink * b)
    return out


if __name__ == '__main__':
    runs = [('bc_s0', {'shrink': 0.0}), ('bc_s0.5', {}), ('bc_s1', {'shrink': 1.0}), ('bc_s0.5_all', {'by': 'all'}),
            ('bc_s0.5_w4', {'window': 4}), ('bc_s1_w4', {'shrink': 1.0, 'window': 4})]
    for nm, kw in runs:
        r = evaluate(bc_fn, **kw)
        report(r, nm, seg=(nm in ('bc_s0', 'bc_s0.5')))
        save_pred(r, nm)

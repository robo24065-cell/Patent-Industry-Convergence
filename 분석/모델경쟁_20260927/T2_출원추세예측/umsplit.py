# CN(및 KR) 실용신안/발명 분리 예측: 두 성분을 따로 외삽해 합산 (CN 규모 효과·실용신안 급증 분리)
import sys, warnings
import numpy as np, pandas as pd
from common import *
from stat_models import logdrift_fn

warnings.filterwarnings('ignore')
ALL = load_all()
UM = ALL['n_um'].values


def um_fn(hist, o, k=8, phi=0.8, phi_um=None):
    T_ = hist.shape[1]
    um = UM[:, :T_]; inv = hist - um
    F = logdrift_fn(hist, o, k=k, phi=phi)
    has = um[:, -k - 1:].min(1) >= 20  # 원점 기준 최근 k+1년 실용신안이 충분한 계열만 분리
    Fi = logdrift_fn(inv, o, k=k, phi=phi)
    Fu = logdrift_fn(um, o, k=k, phi=phi if phi_um is None else phi_um)
    F[has] = (Fi + Fu)[has]
    return F


if __name__ == '__main__':
    for nm, kw in [('um_split', {}), ('um_split_pu6', {'phi_um': 0.6}), ('um_split_k5', {'k': 5})]:
        r = evaluate(um_fn, **kw)
        report(r, nm, seg=True)
        save_pred(r, nm)

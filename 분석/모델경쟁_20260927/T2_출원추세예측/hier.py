# 계층 구조 활용: 국가합(5) / 산업합(50) / 산업×국가(250)
import sys, json, warnings
import numpy as np, pandas as pd
from common import *

warnings.filterwarnings('ignore')
MAIN = load_panel()
CCI = np.array([CCS.index(c) for c in MAIN.index.get_level_values('cc')])
KS = sorted(set(MAIN.index.get_level_values('ksic')))
KSI = np.array([KS.index(k) for k in MAIN.index.get_level_values('ksic')])


def ldrift(M, k, phi):
    L = np.log(np.maximum(M, 1e-9))
    g = (L[:, -1] - L[:, -1 - k]) / k
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.exp(L[:, -1:] + g[:, None] * steps[None, :]), g


def ctot(hist):
    return np.stack([hist[CCI == j].sum(0) for j in range(5)])


def topdown_fn(hist, o, kt=8, pt=0.8, ks=8, ps=0.8):
    """국가합은 로그드리프트, 산업점유율은 로그드리프트(감쇠) → 곱"""
    T = ctot(hist)
    FT, _ = ldrift(T, kt, pt)
    share = (hist + 0.5) / T[CCI]
    FS, _ = ldrift(share, ks, ps)
    # 점유율 합 1로 재정규화
    FSn = FS.copy()
    for j in range(5):
        FSn[CCI == j] /= FS[CCI == j].sum(0, keepdims=True)
    return FT[CCI] * FSn


def shrink_fn(hist, o, k=8, phi=0.8, lam=0.5, kc=8, target='ctot'):
    """자기 로그성장률을 국가합(또는 산업합) 성장률 쪽으로 수축"""
    L = np.log1p(hist)
    g = (L[:, -1] - L[:, -1 - k]) / k
    if target == 'ctot':
        Tt = np.log(ctot(hist)); gc = ((Tt[:, -1] - Tt[:, -1 - kc]) / kc)[CCI]
    else:
        kt = np.stack([hist[KSI == i].sum(0) for i in range(len(KS))]); Lk = np.log(kt)
        gc = ((Lk[:, -1] - Lk[:, -1 - kc]) / kc)[KSI]
    gg = lam * g + (1 - lam) * gc
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return np.expm1(L[:, -1:] + gg[:, None] * steps[None, :])


def recon_fn(hist, o, base='logdrift', kt=8, pt=0.8, alpha=1.0, **kw):
    """하위 예측을 국가합 예측에 비례 조정 (alpha=1 완전 조정, 0 무조정)"""
    from stat_models import logdrift_fn
    F = logdrift_fn(hist, o, **(kw or {'k': 8, 'phi': 0.8}))
    FT, _ = ldrift(ctot(hist), kt, pt)
    Fn = F.copy()
    for j in range(5):
        s = F[CCI == j].sum(0)
        Fn[CCI == j] = F[CCI == j] * (FT[j] / s) ** alpha
    return Fn


if __name__ == '__main__':
    base = summarize(naive_ref())['MASE']
    jobs = [('topdown', topdown_fn, {})]
    for kt, pt in [(3, 1.0), (5, 0.8), (8, 0.8), (8, 1.0)]:
        jobs.append((f'topdown_T{kt}_{pt}', topdown_fn, {'kt': kt, 'pt': pt}))
    for ks, ps in [(5, 0.8), (8, 0.6), (8, 1.0)]:
        jobs.append((f'topdown_S{ks}_{ps}', topdown_fn, {'ks': ks, 'ps': ps}))
    for lam in [0.25, 0.5, 0.75]:
        jobs.append((f'shrink_ctot_l{lam}', shrink_fn, {'lam': lam}))
        jobs.append((f'shrink_ktot_l{lam}', shrink_fn, {'lam': lam, 'target': 'ktot'}))
    for kt, pt in [(8, 0.8), (5, 1.0), (3, 1.0)]:
        jobs.append((f'recon_T{kt}_{pt}', recon_fn, {'kt': kt, 'pt': pt}))
    sel = sys.argv[1:]
    for nm, fn, kw in jobs:
        if sel and not any(nm.startswith(s) for s in sel):
            continue
        r = evaluate(fn, **kw)
        report(r, nm, seg=False)
        save_pred(r, nm)

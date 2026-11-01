# -*- coding: utf-8 -*-
# PREREG 2.1 단일 단절 검정
import numpy as np
YEARS = list(range(2010, 2023))


def _ssr(x):
    return float(((x - x.mean()) ** 2).sum())


def break_test(s, years=YEARS, bset=range(2013, 2021), B=2000, seed=20261013, block=2):
    y = np.asarray([s.get(t, np.nan) for t in years], float)
    if np.isnan(y).any() or len(y) < 8:
        return None
    yrs = np.asarray(years)
    best = None
    for b in bset:
        a, c = y[yrs < b], y[yrs >= b]
        if len(a) < 3 or len(c) < 3:
            continue
        ss = _ssr(a) + _ssr(c)
        if best is None or ss < best[1]:
            best = (b, ss)
    b, ss2 = best
    a, c = y[yrs < b], y[yrs >= b]
    d = c.mean() - a.mean()
    r = d / a.mean() if a.mean() != 0 else np.nan
    n = len(y)
    X = np.c_[np.ones(n), yrs - yrs.mean()]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ssl = float(((y - X @ beta) ** 2).sum())
    eps = 1e-12 * max(1.0, float((y ** 2).mean()))
    bic2 = n * np.log(ss2 / n + eps) + 3 * np.log(n)   # 평균 2개 + 단절 위치
    bicl = n * np.log(ssl / n + eps) + 2 * np.log(n)
    rng = np.random.default_rng(seed)

    def mbb(x):
        m = len(x)
        k = int(np.ceil(m / block))
        starts = rng.integers(0, m - block + 1, size=k)
        return np.concatenate([x[s:s + block] for s in starts])[:m]
    ds = np.array([mbb(c).mean() - mbb(a).mean() for _ in range(B)])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    sig = bool((lo > 0 or hi < 0) and abs(r) >= 0.20 and (bicl - bic2) >= 2)
    return dict(b=int(b), mean_before=float(a.mean()), mean_after=float(c.mean()), delta=float(d), rel=float(r),
                ci_lo=float(lo), ci_hi=float(hi), bic_two_mean=float(bic2), bic_linear=float(bicl), dbic=float(bicl - bic2), sig=sig)

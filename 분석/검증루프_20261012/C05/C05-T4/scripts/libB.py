# 기업키 군집 부트스트랩용 빠른 가중 지표(정렬·동점 묶음을 미리 계산).
import numpy as np


class Prep:
    def __init__(self, y, s):
        y = np.asarray(y).astype(bool); s = np.asarray(s, float)
        self.oa = np.argsort(s, kind='mergesort')
        sa = s[self.oa]
        _, start = np.unique(sa, return_index=True)
        self.grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(sa))))
        self.ng = len(start)
        self.ya = y[self.oa]
        self.od = np.argsort(-s, kind='mergesort'); self.yd = y[self.od].astype(float)

    def auc(self, w):
        wa = w[self.oa]
        g0 = np.bincount(self.grp, weights=np.where(self.ya, 0.0, wa), minlength=self.ng)
        g1 = np.bincount(self.grp, weights=np.where(self.ya, wa, 0.0), minlength=self.ng)
        below = np.cumsum(g0) - g0
        W1, W0 = g1.sum(), g0.sum()
        return float(np.sum(g1 * (below + 0.5 * g0)) / (W1 * W0))

    def ap_top(self, w, k=1000):
        wd = w[self.od]
        tp = np.cumsum(wd * self.yd); n = np.cumsum(wd)
        P = tp[-1]
        ap = float(np.sum((tp / np.maximum(n, 1e-12)) * wd * self.yd) / P) if P > 0 else np.nan
        i = min(np.searchsorted(n, k, side='left'), len(n) - 1)
        return ap, float(tp[i] / n[i])


def boot_weights(keys_codes, n_keys, rng):
    draw = rng.integers(0, n_keys, n_keys)
    cnt = np.bincount(draw, minlength=n_keys).astype(float)
    return cnt[keys_codes]

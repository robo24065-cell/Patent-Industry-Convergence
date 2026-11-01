# 가중 AUC·군집 부트스트랩 공용
import numpy as np, pandas as pd


class AUCPrep:
    """점수 정렬을 한 번만 하고, 행 가중치만 바꿔 AUC를 O(n)으로 다시 계산."""
    def __init__(self, s, y):
        s = np.asarray(s, np.float64); y = np.asarray(y).astype(bool)
        u, inv = np.unique(s, return_inverse=True)
        self.g = inv; self.G = len(u); self.y = y

    def auc(self, w=None):
        if w is None:
            w = np.ones(len(self.y))
        wp = np.bincount(self.g, weights=w * self.y, minlength=self.G)
        wn = np.bincount(self.g, weights=w * (~self.y), minlength=self.G)
        cn = np.cumsum(wn) - wn
        P, N = wp.sum(), wn.sum()
        if P <= 0 or N <= 0:
            return np.nan
        return float((wp * (cn + 0.5 * wn)).sum() / (P * N))


def skill(a1, a0):
    return (a1 - a0) / (1 - a0)


def boot_weights(n_cl, B, seed, kind='poisson'):
    rng = np.random.default_rng(seed)
    return rng.poisson(1.0, size=(B, n_cl)).astype(np.float64)


def cluster_boot(origins, B=200, seed=20261013, twoway=False):
    """origins: list of dict(prep1, prep0, firm_idx(int codes 공통), k_idx, base_w).
    반환: 관측 Δskill(원점 평균), 부트스트랩 배열."""
    obs = np.mean([skill(o['p1'].auc(o['w']), o['p0'].auc(o['w'])) for o in origins])
    nf = max(int(o['f'].max()) for o in origins) + 1
    Wf = boot_weights(nf, B, seed)
    Wk = boot_weights(50, B, seed + 7) if twoway else None
    res = np.empty(B)
    for b in range(B):
        vals = []
        for o in origins:
            w = o['w'] * Wf[b, o['f']]
            if twoway:
                w = w * Wk[b, o['k']]
            vals.append(skill(o['p1'].auc(w), o['p0'].auc(w)))
        res[b] = np.nanmean(vals)
    return obs, res


def cluster_boot_diff(origins, B=200, seed=20261013, twoway=False):
    """AUC 차(p1 − p0) 원점 평균."""
    obs = np.mean([o['p1'].auc(o['w']) - o['p0'].auc(o['w']) for o in origins])
    nf = max(int(o['f'].max()) for o in origins) + 1
    Wf = boot_weights(nf, B, seed)
    Wk = boot_weights(50, B, seed + 7) if twoway else None
    res = np.empty(B)
    for b in range(B):
        vals = []
        for o in origins:
            w = o['w'] * Wf[b, o['f']]
            if twoway:
                w = w * Wk[b, o['k']]
            vals.append(o['p1'].auc(w) - o['p0'].auc(w))
        res[b] = np.nanmean(vals)
    return obs, res

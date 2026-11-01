# 가중 AUC·Δskill·군집 부트스트랩(기업 (cc,aid), 서브클래스 s 공통, 이원 = 곱)
import numpy as np, pandas as pd
from lib12 import *


class AUCPrep:
    """점수 하나에 대해 정렬·동률 그룹을 미리 만들어 두고, 가중치만 바꿔 AUC를 빠르게 계산."""
    def __init__(self, score, y):
        u, g = np.unique(score, return_inverse=True)
        self.g = g; self.ng = len(u); self.y = y.astype(bool)

    def auc(self, w):
        wp = np.bincount(self.g[self.y], weights=w[self.y], minlength=self.ng)
        wn = np.bincount(self.g[~self.y], weights=w[~self.y], minlength=self.ng)
        cn = np.cumsum(wn) - wn
        P = wp.sum(); N = wn.sum()
        return float((wp * (cn + 0.5 * wn)).sum() / (P * N))


def dskill(a, b):
    return (a - b) / (1 - b)


class Boot:
    """여러 원점 예측표(같은 군집 가중 공유)에 대한 부트스트랩 가중 생성기."""
    def __init__(self, frames, B=200, seed=SEED):
        fk = pd.concat([f.cc.astype(str) + ':' + f.aid.astype(str) for f in frames])
        self.fidx = pd.Index(fk.unique())
        self.sidx = pd.Index(np.unique(np.concatenate([f.s.values for f in frames])))
        rf = np.random.default_rng(seed); rs = np.random.default_rng(seed + 7)
        self.WF = rf.poisson(1.0, (B, len(self.fidx))).astype(np.float32)
        self.WS = rs.poisson(1.0, (B, len(self.sidx))).astype(np.float32)
        self.B = B
        self.maps = []
        for f in frames:
            fi = self.fidx.get_indexer(f.cc.astype(str) + ':' + f.aid.astype(str))
            si = self.sidx.get_indexer(f.s.values)
            self.maps.append((fi, si))

    def weights(self, k, b, kind):
        fi, si = self.maps[k]
        if kind == 'firm':
            return self.WF[b, fi]
        if kind == 'sc':
            return self.WS[b, si]
        return self.WF[b, fi] * self.WS[b, si]


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x)
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]

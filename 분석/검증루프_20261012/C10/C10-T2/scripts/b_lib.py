# -*- coding: utf-8 -*-
r"""부트스트랩 커널: 가중 AUC = Σ_pos w_p (W_neg(아래) + 0.5 W_neg(동점)) / (Σw_p Σw_n). c_pool fast()와 수학적으로 같음(검증은 b_selftest)."""
import sys
sys.dont_write_bytecode = True
import numpy as np


class AucPrep:
    def __init__(self, y, s, cl):
        """y 0/1, s 점수, cl: 군집 인덱스 배열 목록(list of int arrays, 행 길이) — 곱 가중 이원 군집이면 2개."""
        y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=np.float64)
        neg = ~y
        sn = s[neg]; o = np.argsort(sn, kind="mergesort"); sn = sn[o]
        sp_ = s[y]
        self.lo = np.searchsorted(sn, sp_, side="left"); self.hi = np.searchsorted(sn, sp_, side="right")
        self.neg_cl = [np.asarray(c)[neg][o].astype(np.int32) for c in cl]
        self.pos_cl = [np.asarray(c)[y].astype(np.int32) for c in cl]

    def auc(self, wcs):
        """wcs: 군집 가중 배열 목록(cl과 같은 순서)."""
        wn = wcs[0][self.neg_cl[0]]
        wp = wcs[0][self.pos_cl[0]]
        for k in range(1, len(wcs)):
            wn = wn * wcs[k][self.neg_cl[k]]; wp = wp * wcs[k][self.pos_cl[k]]
        cum = np.empty(len(wn) + 1); cum[0] = 0; np.cumsum(wn, out=cum[1:])
        below = cum[self.lo]; tied = cum[self.hi] - below
        return float((wp * (below + 0.5 * tied)).sum() / (wp.sum() * cum[-1]))

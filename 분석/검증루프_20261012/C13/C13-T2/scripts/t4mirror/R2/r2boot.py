# -*- coding: utf-8 -*-
"""가중(군집 재표집 횟수) AUC·PR-AUC — 1라운드 final_run 과 같은 계산"""
import numpy as np


def wauc_prep(y, s):
    o = np.argsort(s, kind='mergesort'); so = s[o]
    grp = np.r_[0, np.cumsum(so[1:] != so[:-1])]
    return o, y[o].astype(float), grp


def wauc(pr, w):
    o, yo, grp = pr
    wo = w[o]; G = grp[-1] + 1
    P = np.bincount(grp, wo * yo, G); N = np.bincount(grp, wo * (1 - yo), G)
    below = np.cumsum(N) - N
    return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


def wap(pr, w):
    o, yo, grp = pr
    wo = w[o]; G = grp[-1] + 1
    P = np.bincount(grp, wo * yo, G)[::-1]; N = np.bincount(grp, wo * (1 - yo), G)[::-1]
    tp = np.cumsum(P); fp = np.cumsum(N)
    rec = tp / tp[-1]; prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))

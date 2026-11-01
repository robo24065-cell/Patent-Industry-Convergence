# -*- coding: utf-8 -*-
r"""verify_1: selftest 일부 반복을 lib14 없이 독립 재구현(sklearn 가중 AUC, scipy rankdata ψ)해 캐시 값과 대조.
사용: python -B v1_regen.py <S1|S2> <lo> <hi>  → verify_1/regen_<S>_<lo>_<hi>.json"""
import os, sys, json
import numpy as np
from scipy.stats import rankdata, t as tdist
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "cache")
B = 200


def gen(S, rng):  # selftest.gen과 같은 난수 순서(재현 목적상 생성 규칙만 동일하게 다시 씀)
    if S == "S1":
        NF, NS = 300, 60
        f, s = np.meshgrid(np.arange(NF), np.arange(NS), indexing="ij"); f = f.ravel(); s = s.ravel()
        keep = rng.random(len(f)) < 0.5; f, s = f[keep], s[keep]
    else:
        NS, FP, RP = 60, 5, 30; NF = NS * FP
        f = np.repeat(np.arange(NF), RP); s = f // FP
    a = rng.normal(0, 0.5, NF); b = rng.normal(0, 0.5, NS)
    p = 1 / (1 + np.exp(-(-2.6 + a[f] + b[s])))
    y = (rng.random(len(f)) < p).astype(np.int8)
    eta = 1.2 * y + rng.normal(0, 1, len(f))
    s1 = eta + rng.normal(0, 0.6, len(f))
    u = rng.normal(0, 0.45, NF); v = rng.normal(0, 0.45, NS)
    s2 = eta + rng.normal(0, 1, len(f)) * 0.6 * np.exp(u[f] + v[s])
    _, cell = np.unique(f * NS + s, return_inverse=True)
    return y, s1, s2, f, s, cell, NF, NS, int(cell.max() + 1)


def psi_rank(y, s):
    y = y.astype(bool); n1 = y.sum(); n0 = (~y).sum()
    r_all = rankdata(s); r_pos = rankdata(s[y]); r_neg = rankdata(s[~y])
    A = (r_all[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    v10 = (r_all[y] - r_pos) / n0          # 양성 i보다 낮은 음성 수(+0.5동점)/n0
    v01 = 1 - (r_all[~y] - r_neg) / n1     # 음성 j보다 높은 양성(+0.5동점)/n1
    psi = np.empty(len(s)); psi[y] = (v10 - A) / n1; psi[~y] = (v01 - A) / n0
    return A, psi


def vg(psi, g):
    S = np.bincount(g, weights=psi); S = S[np.bincount(g) > 0]; G = len(S)
    return G / (G - 1) * (S ** 2).sum(), G


def one(S, r):
    rng = np.random.default_rng([777, 1 if S == "S1" else 2, r])
    y, s1, s2, f, s, cell, NF, NS, NC = gen(S, rng)
    A1 = roc_auc_score(y, s1); A2 = roc_auc_score(y, s2)
    a1, p1 = psi_rank(y, s1); a2, p2 = psi_rank(y, s2)
    psi = p1 - p2
    vA, GA = vg(psi, f); vB, GB = vg(psi, s); vAB, _ = vg(psi, cell)
    v3 = vA + vB - vAB
    if v3 <= 0: v3 = max(vA, vB)
    brng = np.random.default_rng([778, 1 if S == "S1" else 2, r])
    e1 = np.empty(B); e2 = np.empty(B)
    for bb in range(B):
        wf = brng.poisson(1.0, NF).astype(float); ws = brng.poisson(1.0, NS).astype(float); wc = brng.poisson(1.0, NC).astype(float)
        w1 = wf[f]; w2 = wf[f] * ws[s]
        m1 = w1 > 0; m2 = w2 > 0
        e1[bb] = roc_auc_score(y[m1], s1[m1], sample_weight=w1[m1]) - roc_auc_score(y[m1], s2[m1], sample_weight=w1[m1])
        e2[bb] = roc_auc_score(y[m2], s1[m2], sample_weight=w2[m2]) - roc_auc_score(y[m2], s2[m2], sample_weight=w2[m2])
    return {"pt": A1 - A2, "pt_rank": a1 - a2, "E3_se": float(np.sqrt(v3)), "E1lin_se": float(np.sqrt(vA)),
            "E1_lo": float(np.percentile(e1, 2.5)), "E1_hi": float(np.percentile(e1, 97.5)), "E1_se": float(e1.std(ddof=1)),
            "E2_lo": float(np.percentile(e2, 2.5)), "E2_hi": float(np.percentile(e2, 97.5)), "E2_se": float(e2.std(ddof=1)),
            "vA": vA, "vB": vB, "vAB": vAB}


if __name__ == "__main__":
    S, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    z = dict(np.load(os.path.join(CACHE, f"st_{S}_0_125.npz")))
    res = []; mx = {}
    for r in range(lo, hi):
        o = one(S, r); res.append(o)
        for k, v in o.items():
            kk = "pt" if k == "pt_rank" else k
            dif = abs(v - z[kk][r]); mx[k] = max(mx.get(k, 0), float(dif))
        print(S, r, flush=True)
    json.dump({"S": S, "reps": [lo, hi], "max_abs_diff_vs_cache": mx}, open(os.path.join(HERE, f"regen_{S}_{lo}_{hi}.json"), "w"), indent=1)
    print(mx)

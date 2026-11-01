# -*- coding: utf-8 -*-
r"""verify_1: 'E4 탈락은 B=200 잡음' 점검. 같은 합성 반복에서 E4를 B=1000(독립 난수 779)으로 다시 계산해
반복 간 SE 변동계수와 평균 SE/참SE가 E3 쪽으로 가는지 본다. 사용: python -B v1_e4B.py <S> <lo> <hi>"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)
from v1_regen import gen  # noqa  (같은 폴더)
from lib14 import AucPrep, placement, vclust  # 읽기 전용 import(쓰기 없음)
sys.path.insert(0, HERE)
BB = int(os.environ.get("BB", "1000"))


def one(S, r):
    rng = np.random.default_rng([777, 1 if S == "S1" else 2, r])
    y, s1, s2, f, s, cell, NF, NS, NC = gen(S, rng)
    A1, p1 = placement(y, s1); A2, p2 = placement(y, s2)
    psi = p1 - p2
    vA, _ = vclust(psi, f, NF); vB, _ = vclust(psi, s, NS); vAB, _ = vclust(psi, cell, NC)
    P = [AucPrep(y, x, [f, s, cell]) for x in (s1, s2)]
    brng = np.random.default_rng([779, 1 if S == "S1" else 2, r])
    e2 = np.empty(BB); ec = np.empty(BB)
    oF, oS, oC = np.ones(NF), np.ones(NS), np.ones(NC)
    for b in range(BB):
        wf = brng.poisson(1.0, NF).astype(float); ws = brng.poisson(1.0, NS).astype(float); wc = brng.poisson(1.0, NC).astype(float)
        e2[b] = P[0].auc([wf, ws, oC]) - P[1].auc([wf, ws, oC])
        ec[b] = P[0].auc([oF, oS, wc]) - P[1].auc([oF, oS, wc])
    v4 = e2.var(ddof=1) - 2 * ec.var(ddof=1)
    fix = v4 <= 0
    if fix: v4 = max(vA, vB)
    return A1 - A2, np.sqrt(vA + vB - vAB), np.sqrt(v4), fix


if __name__ == "__main__":
    S, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    R = np.array([one(S, r) for r in range(lo, hi)], dtype=float)
    np.save(os.path.join(HERE, f"e4B{BB}_{S}_{lo}_{hi}.npy"), R)
    print(S, lo, hi, BB, "done", flush=True)

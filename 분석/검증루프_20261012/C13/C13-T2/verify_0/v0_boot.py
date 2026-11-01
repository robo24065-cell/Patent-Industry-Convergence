# -*- coding: utf-8 -*-
"""verify_0 독립 CI 재계산(N4·N7). 트랙 cluster_boot 코드를 쓰지 않고 같은 난수 규약(np.unique 정렬 군집, rng.integers로 군집 복원추출)을
'군집 뽑힌 횟수 = 행 가중치' + 가중 Mann–Whitney AUC(동점 0.5)로 다시 구현해, 공식 묶음과 CI 하한이 가장 낮은 묶음의 CI를 대조한다(허용 1e−9).
→ verify_0/v0_boot.json"""
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V); C = os.path.join(W, "cache")


def wauc(y, p, w):
    o = np.argsort(p, kind="mergesort"); ps, ys, ws = p[o], y[o], w[o]
    g = np.r_[0, np.cumsum(ps[1:] != ps[:-1])]; G = g[-1] + 1
    P = np.bincount(g, ws * ys, G); N = np.bincount(g, ws * (1 - ys), G)
    below = np.cumsum(N) - N
    return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


def boot(y, pm, pb, cl, n, seed):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True)
    S = []
    for _ in range(n):
        cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        w = cnt[inv]
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            continue
        a1, a0 = wauc(y, pm, w), wauc(y, pb, w)
        S.append((a1 - a0) / (1 - a0))
    return [float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))]


out = {}
for t, strong, seeds in (("n4", "p_M0ens", (2023, 2024)), ("n7", "p_SB", (2022, 2023))):
    bs = []
    for f in os.listdir(os.path.join(C, t)):
        if f.startswith("b") and f.endswith(".json"):
            bs.append(json.load(open(os.path.join(C, t, f), encoding="utf-8")))
    pick = sorted({0, min(bs, key=lambda r: r["ci_lo"])["bundle"], min(bs, key=lambda r: r["strong_ci_lo"])["bundle"]})
    for b in pick:
        J = [r for r in bs if r["bundle"] == b][0]
        P = pd.read_parquet(os.path.join(C, t, f"preds_b{b}.parquet"))
        y = P.y.values.astype(float); cl = P.bizno.values
        ci = boot(y, P.p_champ.values, P.p_M0.values, cl, 1000, seeds[0])
        cs = boot(y, P.p_champ.values, P[strong].values, cl, 1000, seeds[1])
        d = max(abs(ci[0] - J["ci_lo"]), abs(ci[1] - J["ci_hi"]), abs(cs[0] - J["strong_ci_lo"]), abs(cs[1] - J["strong_ci_hi"]))
        out[f"{t}_b{b}"] = dict(ci_indep=ci, ci_json=[J["ci_lo"], J["ci_hi"]], strong_ci_indep=cs, strong_ci_json=[J["strong_ci_lo"], J["strong_ci_hi"]],
                               max_abs_diff=d, pass_1e9=d <= 1e-9)
        print(t, b, out[f"{t}_b{b}"], flush=True)
json.dump(out, open(os.path.join(V, "v0_boot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

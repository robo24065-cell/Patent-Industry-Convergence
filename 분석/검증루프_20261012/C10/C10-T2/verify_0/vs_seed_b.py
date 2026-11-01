# -*- coding: utf-8 -*-
r"""반박 검증(seed 견고성) B: 작업자와 다른 방식의 법인 군집 CI.
법인 다항 재표집(개수 가중, Poisson 아님), 다른 난수(rng 9090), dense-rank bincount AUC.
사용: python -B vs_seed_b.py <k,p;k,p;...> <B> <tag>  → verify_0/vs_seed_b_<tag>.json
P50-FREE50 5원점 평균 ΔAUC의 백분위 95% CI (원점 간 같은 법인 가중 공유)."""
import os, sys, json, time, glob
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
NC8 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(W))), "모델경쟁_20260927", "T1_산업대기실", "NONCIRC_R8", "cache")
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
CS = [2008, 2009, 2010, 2011, 2012]
cfgs = [tuple(int(x) for x in s.split(",")) for s in sys.argv[1].split(";")]
B = int(sys.argv[2]); tag = sys.argv[3]
A = {c: pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["applicantCd", "first_ky"]) for c in CS}
firms = pd.unique(pd.concat([A[c].applicantCd for c in CS])); fidx = {f: i for i, f in enumerate(firms)}
G = {c: A[c].applicantCd.map(fidx).values for c in CS}; Y = {c: A[c].first_ky.notna().values for c in CS}


def getcol(c, k, col):
    for f in sorted(glob.glob(os.path.join(W, "cache", f"pred_c{c}_k{k}_P50-FREE50-TXT50_p*.parquet")), key=len, reverse=True):
        d = pd.read_parquet(f)
        if col in d.columns:
            return d[col].values.astype(np.float64)
    raise KeyError(col)


def wauc(r, nu, y, w):
    wp = np.bincount(r[y], weights=w[y], minlength=nu); wn = np.bincount(r[~y], weights=w[~y], minlength=nu)
    below = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


PR = {}
for (k, p) in cfgs:
    for c in CS:
        for arm in ["P50", "FREE50"]:
            u, r = np.unique(getcol(c, k, f"{arm}_p{p}"), return_inverse=True)
            PR[(k, p, c, arm)] = (r, len(u))
    log("prep", k, p)
rng = np.random.default_rng(9090); nf = len(firms)
pt = {}; bs = {cf: [] for cf in cfgs}
for (k, p) in cfgs:
    one = np.ones(nf)
    pt[(k, p)] = float(np.mean([wauc(*PR[(k, p, c, "P50")], Y[c], one[G[c]]) - wauc(*PR[(k, p, c, "FREE50")], Y[c], one[G[c]]) for c in CS]))
for b in range(B):
    wf = rng.multinomial(nf, np.full(nf, 1.0 / nf)).astype(np.float64)
    for (k, p) in cfgs:
        bs[(k, p)].append(np.mean([wauc(*PR[(k, p, c, "P50")], Y[c], wf[G[c]]) - wauc(*PR[(k, p, c, "FREE50")], Y[c], wf[G[c]]) for c in CS]))
    if b % 50 == 0:
        log("rep", b)
out = {}
for cf in cfgs:
    x = np.array(bs[cf])
    out[f"k{cf[0]}_p{cf[1]}"] = {"point": pt[cf], "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                                  "p_gt0": float((x > 0).mean()), "boot_sd": float(x.std(ddof=1)), "B": B, "n_firm": int(nf)}
json.dump(out, open(os.path.join(HERE, f"vs_seed_b_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", out)

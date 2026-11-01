# -*- coding: utf-8 -*-
r"""verify_1 군집 다항 재표집(작업자 Poisson과 다른 방식, 다른 난수) — 공식 seed(k0,p0) 예측.
사용: python -B v1_boot.py <firm|sc> <B> <seed>
군집 가중은 5원점에 공유. AUC는 정렬+reduceat 가중 Mann–Whitney(동점 0.5). 가중 1이면 sklearn과 같아야 함(점검 기록).
출력: verify_1/v1_boot_<cl>.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
NC8 = os.path.join(BASE, r"분석\모델경쟁_20260927\T1_산업대기실\NONCIRC_R8\cache")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE); C = os.path.join(W, "cache")
CS = [2008, 2009, 2010, 2011, 2012]
cl, B, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
ARMS = ["P50", "FREE50", "FREEIPC50", "HSC1_50", "P50FIPC", "P50HSC1"]
PAIRS = {"FIPC-FREE50": ("FREEIPC50", "FREE50"), "HSC1-FREE50": ("HSC1_50", "FREE50"),
         "P50FIPC-P50": ("P50FIPC", "P50"), "P50HSC1-P50": ("P50HSC1", "P50"),
         "P50FIPC-FIPC": ("P50FIPC", "FREEIPC50"), "P50HSC1-HSC1": ("P50HSC1", "HSC1_50"), "P50-FREE50": ("P50", "FREE50")}
raw = {}
for c in CS:
    a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"])
    P = pd.concat([pd.read_parquet(os.path.join(C, f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0"]),
                   pd.read_parquet(os.path.join(C, f"pred_c{c}_k0_FREEIPC50-HSC1_50-P50HSC1_p0.parquet")),
                   pd.read_parquet(os.path.join(C, f"pred_c{c}_k0_HMG3_50-P50FIPC-P50HMG3_p0.parquet"), columns=["P50FIPC_p0"])], axis=1)
    raw[c] = (a, P)
key = "applicantCd" if cl == "firm" else "sc"
U = np.array(sorted(set().union(*[set(raw[c][0][key].unique()) for c in CS])))
G = len(U)
PR = {}
for c in CS:
    a, P = raw[c]
    g = np.searchsorted(U, a[key].values); assert (U[g] == a[key].values).all()
    y = a.first_ky.notna().values
    for m in ARMS:
        s = P[m + "_p0"].values.astype(np.float64)
        o = np.argsort(s, kind="mergesort"); ss = s[o]
        st = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        PR[(c, m)] = (g[o], y[o], st, roc_auc_score(y, s))
del raw


def wauc(pr, wg):
    go, yo, st, _ = pr
    w = wg[go]
    wp = np.add.reduceat(np.where(yo, w, 0.0), st); wn = np.add.reduceat(np.where(yo, 0.0, w), st)
    below = np.cumsum(wn) - wn
    return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


one = np.ones(G)
chk = max(abs(wauc(PR[k], one) - PR[k][3]) for k in PR)
log("n_clusters", G, "unit-weight vs sklearn maxdiff", chk)
rng = np.random.default_rng(seed)
bd = {k: [] for k in PAIRS}
for r in range(B):
    wg = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
    au = {k: wauc(PR[k], wg) for k in PR}
    for nm, (x, z) in PAIRS.items():
        bd[nm].append(np.mean([au[(c, x)] - au[(c, z)] for c in CS]))
    if r % 50 == 0:
        log("rep", r)
pt = {nm: float(np.mean([PR[(c, x)][3] - PR[(c, z)][3] for c in CS])) for nm, (x, z) in PAIRS.items()}
res = {"cluster": cl, "n_clusters": int(G), "B": B, "seed": seed, "unit_vs_sklearn_maxdiff": float(chk),
       "res": {nm: {"point": pt[nm], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    "p_gt0": float(np.mean(np.array(v) > 0)), "sd": float(np.std(v, ddof=1))} for nm, v in bd.items()}}
json.dump(res, open(os.path.join(HERE, f"v1_boot_{cl}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done", json.dumps({k: [round(v["point"], 5)] + [round(x, 5) for x in v["ci95"]] for k, v in res["res"].items()}))

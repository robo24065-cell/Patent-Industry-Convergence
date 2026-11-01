# -*- coding: utf-8 -*-
r"""verify_0 반박 시도 2: '이원 군집 CI가 0을 포함한 것은 곱 가중 부트스트랩이 지나치게 보수적이라서인가?'
Cameron–Gelbach–Miller 이원 분산 V = V_firm + V_sc − V_(firm∩sc) 를 독립 코드로 추정(각 성분 Poisson 부트스트랩 B=300, dense-rank AUC).
firm∩sc 군집 = (법인, 서브클래스) 쌍(원점 간 공유). 대상: P50−FREE50, P50−HSC1_50. → verify_0/v_result2.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
NC8 = os.path.join(BASE, r"분석\모델경쟁_20260927\T1_산업대기실\NONCIRC_R8\cache")
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
CS = [2008, 2009, 2010, 2011, 2012]; ARM = ["P50", "FREE50", "HSC1_50"]
D = {}
for c in CS:
    a = pd.read_parquet(os.path.join(NC8, f"A_o{c}.parquet"), columns=["applicantCd", "sc", "first_ky"])
    p1 = pd.read_parquet(os.path.join(W, "cache", f"pred_c{c}_k0_P50-FREE50-TXT50_p012.parquet"), columns=["P50_p0", "FREE50_p0"])
    p2 = pd.read_parquet(os.path.join(W, "cache", f"pred_c{c}_k0_FREEIPC50-HSC1_50-P50HSC1_p0.parquet"), columns=["HSC1_50_p0"])
    D[c] = pd.concat([a, p1, p2], axis=1)
allr = pd.concat(D.values())
keys = {"firm": allr.applicantCd, "sc": allr.sc, "pair": allr.applicantCd.astype(str) + "|" + allr.sc}
U = {k: {u: i for i, u in enumerate(pd.unique(v))} for k, v in keys.items()}
PR = {}
for c, d in D.items():
    y = d.first_ky.notna().values
    g = {"firm": d.applicantCd.map(U["firm"]).values, "sc": d.sc.map(U["sc"]).values,
         "pair": (d.applicantCd.astype(str) + "|" + d.sc).map(U["pair"]).values}
    for arm in ARM:
        u, r = np.unique(d[arm + "_p0"].values.astype(np.float64), return_inverse=True)
        PR[(c, arm)] = (r, len(u), y, g)


def wauc(pr, comp, wg):
    r, nu, y, g = pr; w = wg[g[comp]]
    wp = np.bincount(r[y], weights=w[y], minlength=nu); wn = np.bincount(r[~y], weights=w[~y], minlength=nu)
    below = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


B = 300; rng = np.random.default_rng(4242); out = {}
draws = {comp: {b: [] for b in ("FREE50", "HSC1_50")} for comp in ("firm", "sc", "pair")}
for comp in ("firm", "sc", "pair"):
    for r_ in range(B):
        wg = rng.poisson(1.0, len(U[comp])).astype(float)
        au = {(c, arm): wauc(PR[(c, arm)], comp, wg) for c in CS for arm in ARM}
        for b in draws[comp]:
            draws[comp][b].append(np.mean([au[(c, "P50")] - au[(c, b)] for c in CS]))
    log(comp)
for b in ("FREE50", "HSC1_50"):
    pt = float(np.mean([wauc(PR[(c, "P50")], "firm", np.ones(len(U["firm"]))) - wauc(PR[(c, b)], "firm", np.ones(len(U["firm"]))) for c in CS]))
    v = {comp: float(np.var(draws[comp][b], ddof=1)) for comp in draws}
    vcgm = v["firm"] + v["sc"] - v["pair"]
    sd_ub = float(np.sqrt(v["firm"] + v["sc"]))
    out[f"P50-{b}"] = {"point": pt, "var": v, "cgm_var": vcgm, "cgm_ci95_normal": [pt - 1.96 * np.sqrt(max(vcgm, 0)), pt + 1.96 * np.sqrt(max(vcgm, 0))],
                       "ub_ci95_normal(V_firm+V_sc)": [pt - 1.96 * sd_ub, pt + 1.96 * sd_ub], "B": B,
                       "n_clusters": {k: len(U[k]) for k in U}}
json.dump(out, open(os.path.join(HERE, "v_result2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log(json.dumps(out, default=float))

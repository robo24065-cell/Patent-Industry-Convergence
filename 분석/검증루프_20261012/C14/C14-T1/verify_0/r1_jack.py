# -*- coding: utf-8 -*-
r"""verify_0 R1: 이원 군집 잭나이프(정확한 AUC 재계산, 선형화 아님). 다른 코드 경로: 행 0/1 마스크 + 자체 순위 AUC(np.unique 역색인, bincount).
sc(또는 ksic_t) delete-one, 법인 무작위 100묶음, 칸 무작위 100묶음 delete-one-group → V_J = V_A + V_B − V_AB.
→ verify_0/r1_jack.json"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import t as tdist
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
T0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
z = np.load(os.path.join(W, "cache", "data14.npz"))
CS = [2008, 2009, 2010, 2011, 2012]; ARMS = ["P50", "FREE50", "TXT50", "FREEIPC50", "HSC1_50"]
PAIRS = [("P50", "FREE50"), ("P50", "FREEIPC50"), ("P50", "HSC1_50"), ("P50", "TXT50")]
G = {k: int(z[f"G_{k}"]) for k in ("firm", "sc", "ksic_t", "cell_fs", "cell_fk")}
PR = {}
for c in CS:
    y = z[f"y_{c}"].astype(bool)
    for a in ARMS:
        u, r = np.unique(z[f"{a}_{c}"], return_inverse=True)
        PR[(c, a)] = (r, len(u), y)


def auc_mask(c, a, keep):
    r, nu, y = PR[(c, a)]
    wp = np.bincount(r[y & keep], minlength=nu).astype(float); wn = np.bincount(r[~y & keep], minlength=nu).astype(float)
    below = np.cumsum(wn) - wn
    return (wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())


def theta(keepf):
    """keepf(c) → 행 마스크. 반환 pair→5원점 평균 Δ"""
    au = {}
    for c in CS:
        k = keepf(c)
        for a in ARMS:
            au[(c, a)] = auc_mask(c, a, k)
    return {f"{a}-{b}": float(np.mean([au[(c, a)] - au[(c, b)] for c in CS])) for a, b in PAIRS}


full = theta(lambda c: np.ones(len(z[f"y_{c}"]), bool))
rng = np.random.default_rng(31415)
grp = {"firm": rng.integers(0, 100, G["firm"]), "cell_fs": rng.integers(0, 100, G["cell_fs"]), "cell_fk": rng.integers(0, 100, G["cell_fk"]),
       "sc": np.arange(G["sc"]), "ksic_t": np.arange(G["ksic_t"])}
JV = {}
ONLY = sys.argv[1:] or ["sc", "ksic_t", "firm", "cell_fs", "cell_fk"]
for comp in ONLY:
    gid = {c: grp[comp][z[f"{comp}_{c}"]] for c in CS}
    ng = int(grp[comp].max() + 1)
    th = []
    for g in range(ng):
        th.append(theta(lambda c: gid[c] != g))
    JV[comp] = {}
    for p in full:
        v = np.array([t[p] for t in th]); JV[comp][p] = float((ng - 1) / ng * ((v - v.mean()) ** 2).sum())
    JV[comp]["_G"] = ng
    log(comp, ng, {p: JV[comp][p] for p in full})
    json.dump({"full": full, "jv": JV[comp]}, open(os.path.join(HERE, f"r1_jv_{comp}.json"), "w", encoding="utf-8"), indent=1)
JV = {}
for comp in ("sc", "ksic_t", "firm", "cell_fs", "cell_fk"):
    f = os.path.join(HERE, f"r1_jv_{comp}.json")
    if not os.path.exists(f):
        log("partial; missing", comp); sys.exit(0)
    jj = json.load(open(f, encoding="utf-8")); JV[comp] = jj["jv"]
    assert all(abs(jj["full"][p] - full[p]) < 1e-12 for p in full)
out = {"full": full, "jack_var": JV}
for s, (A_, B_, AB_) in {"fs": ("firm", "sc", "cell_fs"), "fk": ("firm", "ksic_t", "cell_fk")}.items():
    out[s] = {}
    for p in full:
        v = JV[A_][p] + JV[B_][p] - JV[AB_][p]; fixed = v <= 0
        if fixed: v = max(JV[A_][p], JV[B_][p])
        df = min(JV[A_]["_G"], JV[B_]["_G"]) - 1; q = tdist.ppf(0.975, df); se = float(np.sqrt(v))
        out[s][p] = {"v": v, "se": se, "lo": full[p] - q * se, "hi": full[p] + q * se, "df": df, "fixed": bool(fixed),
                     "pigeon_like_se": float(np.sqrt(JV[A_][p] + JV[B_][p] + JV[AB_][p]))}
json.dump(out, open(os.path.join(HERE, "r1_jack.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log(json.dumps({s: {p: (round(v["lo"], 6), round(v["hi"], 6)) for p, v in out[s].items()} for s in ("fs", "fk")}))

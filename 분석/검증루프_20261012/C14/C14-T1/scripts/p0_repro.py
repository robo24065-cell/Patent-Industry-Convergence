# -*- coding: utf-8 -*-
r"""P0: C10-T2 E1·E2 재현. (1) cache npz 재집계 vs boot_summary.csv, (2) 가중 난수 규칙으로 rep 0–39 재계산 vs npz.
→ res/p0_repro.json, cache/boot_c10_all.npz(5팔만, E2·E1 분산 계산용)"""
import sys, glob
sys.dont_write_bytecode = True
from lib14 import *

CLT = {"firm": 1, "ksic_t": 2, "ksic_f": 3, "sc": 4, "firm_sc": 5, "firm_ksic_t": 6}
COMPS = {"firm": ["firm"], "firm_sc": ["firm", "sc"], "firm_ksic_t": ["firm", "ksic_t"], "sc": ["sc"], "ksic_t": ["ksic_t"]}
D, G = build_data()
log("data", G)
BS = pd.read_csv(os.path.join(C10, "res", "boot_summary.csv"))
out = {"G": G, "agg": {}, "rerun": {}}
ALL = {}
for clt in ("firm", "firm_sc", "firm_ksic_t", "sc", "ksic_t"):
    fs = glob.glob(os.path.join(C10C, f"boot_{clt}_*.npz"))
    fs = [f for f in fs if os.path.basename(f)[5:-4].rsplit("_", 2)[0] == clt]
    parts = sorted([(int(os.path.basename(f)[:-4].split("_")[-2]), np.load(f)) for f in fs], key=lambda x: x[0])
    arms = list(parts[0][1]["arms"]); A = np.concatenate([p[1]["auc"] for p in parts])
    assert [p[0] for p in parts] == list(range(0, 2000, 250)), [p[0] for p in parts]
    ALL[clt] = A[:, [arms.index(a) for a in ARMS], :]
    for a, b in PAIRS:
        db = (A[:, arms.index(a), :] - A[:, arms.index(b), :]).mean(1)
        r = BS[BS.pair == f"{a}-{b}"].iloc[0]
        lo, hi = np.percentile(db, 2.5), np.percentile(db, 97.5)
        out["agg"][f"{clt}|{a}-{b}"] = {"lo": lo, "hi": hi, "c10_lo": float(r[f"{clt}_lo"]), "c10_hi": float(r[f"{clt}_hi"]),
                                         "absdiff": float(max(abs(lo - r[f"{clt}_lo"]), abs(hi - r[f"{clt}_hi"]))),
                                         "boot_sd": float(np.std(db, ddof=1)), "boot_mean": float(db.mean()), "B": len(db)}
np.savez_compressed(safe_path(os.path.join(CACHE, "boot_c10_all.npz")), **ALL, arms=np.array(ARMS))
log("agg done")
# (2) 재계산
for clt in ("firm", "firm_sc"):
    comps = COMPS[clt]
    PR = {}
    for j, c in enumerate(C_ORIG):
        for i, arm in enumerate(ARMS):
            PR[(i, j)] = AucPrep(D[c]["y"], D[c][arm], [D[c][comp] for comp in comps])
    mx = 0.0
    for r in range(40):
        ws = [np.random.default_rng([20261013, CLT[clt], q, r]).poisson(1.0, G[comp]).astype(np.float64) for q, comp in enumerate(comps)]
        for (i, j), P_ in PR.items():
            mx = max(mx, abs(P_.auc(ws) - ALL[clt][r, i, j]))
    out["rerun"][clt] = {"reps": 40, "max_absdiff": mx}
    log("rerun", clt, mx)
out["pass_agg"] = all(v["absdiff"] <= 1e-6 for v in out["agg"].values())
out["pass_rerun"] = all(v["max_absdiff"] <= 1e-9 for v in out["rerun"].values())
# 점추정
pt = {}
for a, b in PAIRS:
    from lib14 import placement
    pt[f"{a}-{b}"] = float(np.mean([placement(D[c]["y"], D[c][a])[0] - placement(D[c]["y"], D[c][b])[0] for c in C_ORIG]))
out["point"] = pt
json.dump(out, open(safe_path(os.path.join(RES, "p0_repro.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
log("pass", out["pass_agg"], out["pass_rerun"], pt)

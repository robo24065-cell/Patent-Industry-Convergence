# -*- coding: utf-8 -*-
r"""E4 성분 부트스트랩. mode: cell_fs | cell_fk (칸 Poisson) | add_fs | add_fk (가산 가중 (w_A+w_B)/2).
사용: python -B e4_boot.py <mode> <lo> <hi> → cache/e4_<mode>_<lo>_<hi>.npz (auc[rep, arm, origin])"""
import sys
sys.dont_write_bytecode = True
from lib14 import *

MODE = {"cell_fs": (1, ["cell_fs"]), "cell_fk": (2, ["cell_fk"]), "add_fs": (3, ["firm", "sc"]), "add_fk": (4, ["firm", "ksic_t"])}
if __name__ == "__main__":
    mode, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    mid, comps = MODE[mode]
    D, G = build_data()
    PR = {}
    for j, c in enumerate(C_ORIG):
        for i, arm in enumerate(ARMS):
            PR[(i, j)] = (AucPrep(D[c]["y"], D[c][arm], [D[c][comp] for comp in comps]))
    AU = np.full((hi - lo, len(ARMS), 5), np.nan)
    for r in range(lo, hi):
        ws = [np.random.default_rng([1417, mid, q, r]).poisson(1.0, G[comp]).astype(np.float64) for q, comp in enumerate(comps)]
        for (i, j), P_ in PR.items():
            if len(comps) == 1:
                AU[r - lo, i, j] = P_.auc(ws)
            else:
                wn = 0.5 * (ws[0][P_.neg_cl[0]] + ws[1][P_.neg_cl[1]]); wp = 0.5 * (ws[0][P_.pos_cl[0]] + ws[1][P_.pos_cl[1]])
                AU[r - lo, i, j] = P_.auc_rowweights(wn, wp)
        if (r - lo) % 100 == 0:
            log("rep", r)
    np.savez_compressed(safe_path(os.path.join(CACHE, f"e4_{mode}_{lo}_{hi}.npz")), auc=AU, arms=np.array(ARMS))
    log("done", mode, lo, hi)

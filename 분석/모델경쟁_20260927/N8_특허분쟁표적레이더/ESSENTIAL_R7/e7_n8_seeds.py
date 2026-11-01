# -*- coding: utf-8 -*-
"""N8 R7 민감도(INFO): 챔피언의 음성 5% 표본 시드만 1·2로 바꿔 같은 변형을 재학습 → 변형−챔피언 AUC 차이가 시드에 견고한지. 공식 비교는 시드 0(e7_n8.py)."""
import os, sys, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
sys.path.insert(0, TR)
import numpy as np, pandas as pd
import n8lib as L
from e_run import score, M0
SPEC = json.load(open(os.path.join(TR, "final_spec_r6.json"), encoding="utf-8"))
FE = SPEC["champion"]["feats"]; CFG = {"neg_frac": SPEC["champion"]["neg_frac"]}
KS = ["ksic_i", "ksic_disp5"]; FAM = ["fam_offices", "fam_us"]; SX = {"ksic_i": "ksic_i_x", "ksic_disp5": "ksic_disp5_x"}
V = {"CH": FE, "R_KSIC": [f for f in FE if f not in KS], "R_FAM": [f for f in FE if f not in FAM],
     "R_PROV": [f for f in FE if f not in KS + FAM], "S_KSIC": [SX.get(f, f) for f in FE], "M0": M0}
D = pd.read_parquet(os.path.join(TR, "cache", "panel_dev.parquet"), columns=list(dict.fromkeys(["rgst_no", "t0", "y"] + FE + M0)))
D = D.merge(pd.read_parquet(os.path.join(HERE, "ksic_x_dev.parquet"), columns=["rgst_no", "t0", "ksic_i_x", "ksic_disp5_x"]), on=["rgst_no", "t0"], how="left")
seed = int(sys.argv[1])
out = {}
for k, f in V.items():
    t = time.time(); P = score(D, f, CFG, "xgb", seed=seed)
    out[k] = {str(v): float(L.auc(D.y.values[D.t0.values == v], P[v])) for v in L.DEV}
    print(seed, k, round(np.mean(list(out[k].values())), 4), round(time.time() - t), flush=True)
json.dump(out, open(os.path.join(HERE, f"seeds_s{seed}.json"), "w", encoding="utf-8"), indent=1)

# -*- coding: utf-8 -*-
"""사후: 음성 표본추출 시드 1로 챔피언 vs M0 재실행(J1·J2, XGB)."""
import json, numpy as np, pandas as pd
from r5_common import *
from r5_eval_spec import SPEC
rows = []
for idea in ["J1", "J2"]:
    S = SPEC[idea]; D = pd.read_parquet(f"{CACHE}/{S['panel']}.parquet").reset_index(drop=True)
    ch = json.load(open(f"{CACHE}/champ_{idea}_xgb.json", encoding="utf-8"))["feats"]
    add = [f for f in ch if f not in S["base"]]
    R, O, _ = run_steps(D, "y", S["dev"], S["base"], [("CH", add)], gap=S["gap"], group_col=None, models=("xgb",), neg_frac=S.get("neg_frac"), n_boot=0 or 50, seed=1)
    rows.append(dict(idea=idea, mean_skill=O.skill.mean(), min_skill=O.skill.min(), win=(O.auc_champ > O.auc_M0).mean(), auc_champ=O.auc_champ.mean(), auc_M0=O.auc_M0.mean()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/R5_seed1.csv", index=False, encoding="utf-8-sig")
